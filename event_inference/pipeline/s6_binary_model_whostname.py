import warnings
import utils
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
import time
from multiprocessing import Pool
import Constants as c


warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

num_pools = 1


default_models = ['rf']
model_list = ['rf']
root_output = ''
root_feature = ''
root_model = ''
mac_dic = {}



# is_error is either 0 or 1
def print_usage(is_error):
    print(c.PREDICT_MOD_USAGE, file=sys.stderr) if is_error else print(c.PREDICT_MOD_USAGE)
    exit(is_error)



def main():
    global  root_output, model_list , root_feature, root_model

    # Parse Arguments
    parser = argparse.ArgumentParser(usage=c.PREDICT_MOD_USAGE, add_help=False)
    parser.add_argument("-i", dest="root_feature", default="")
    parser.add_argument("-o", dest="root_model", default="")
    parser.add_argument("-r", dest="rf", action="store_true", default=False)

    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    if args.help:
        print_usage(0)

    print("Running %s..." % c.PATH)

    # Error checking command line args
    root_feature = args.root_feature
    root_model = args.root_model
    errors = False
    # check -i in features
    if root_feature == "":
        errors = True
        print(c.NO_FEAT_DIR, file=sys.stderr)
    elif not os.path.isdir(root_feature):
        errors = True
        print(c.INVAL % ("Features directory", root_feature, "directory"), file=sys.stderr)
    else:
        if not os.access(root_feature, os.R_OK):
            errors = True
            print(c.NO_PERM % ("features directory", root_feature, "read"), file=sys.stderr)
        if not os.access(root_feature, os.X_OK):
            errors = True
            print(c.NO_PERM % ("features directory", root_feature, "execute"), file=sys.stderr)

    # check -o out models
    if root_model == "":
        errors = True
        print(c.NO_MOD_DIR, file=sys.stderr)
    elif os.path.isdir(root_model):
        if not os.access(root_model, os.W_OK):
            errors = True
            print(c.NO_PERM % ("model directory", root_model, "write"), file=sys.stderr)
        if not os.access(root_model, os.X_OK):
            errors = True
            print(c.NO_PERM % ("model directory", root_model, "execute"), file=sys.stderr)


    if not model_list:
        model_list = default_models.copy()

    if errors:
        print_usage(1)
    # end error checking

    print("Input files located in: %s\nOutput files placed in: %s" % (root_feature, root_model))
    root_output = os.path.join(root_model, 'output')
    if not os.path.exists(root_output):
        os.system('mkdir -pv %s' % root_output)
        for model_alg in model_list:
            model_dir = '%s/%s' % (root_model, model_alg)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

    train_models()


def train_models():
    # global root_feature, root_model, root_output
    """
    Scan feature folder for each device
    """
    print('root_feature: %s' % root_feature)
    print('root_model: %s' % root_model)
    print('root_output: %s' % root_output)
    lfiles = []
    lparas = []
    ldnames = []

    random_state = 422
    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            print(csv_file)
            train_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]
            lparas.append((train_data_file, dname, random_state))
    p = Pool(num_pools)
    t0 = time.time()
    list_results = p.map(eid_wrapper, lparas)
    for ret in list_results:
        if ret is None or len(ret) == 0: continue
        for res in ret:
            tmp_outfile = res[0]
            tmp_res = res[1:]
            with open(tmp_outfile, 'a+') as off:
                # off.write('random_state:',random_state)
                off.write('%s\n' % '\t'.join(map(str, tmp_res)))
                print('Agg saved to %s' % tmp_outfile)
    t1 = time.time()
    print('Time to train all models for %s devices using %s threads: %.2f' % (len(lparas),num_pools, (t1 - t0)))



def eid_wrapper(a):
    """
    wrapper of the training/testing funtion
    INPUT:
        train_data_file, dname, random_state
    RETURN:
        results
    """
    return eval_individual_device(a[0], a[1], a[2])


def eval_individual_device(train_data_file, dname, random_state):
    global root_feature, root_model, root_output

    """
    training/testing/evaluation of an individual device 
    """

    list_models_todo = []
    for model_alg in model_list:
        """
        Prepare the directories and add only models that have not been trained yet 
        """
        model_dir = '%s/%s' % (root_model, model_alg)
        # model_file = '%s/%s%s.model' % (model_dir, dname, model_alg)
        label_file = '%s/%s.label.txt' % (model_dir, dname)

        list_models_todo.append(model_alg)

    print('Training %s using algorithm(s): %s' % (dname, str(list_models_todo)))


    """
    read training data
    """
    train_data = pd.read_csv(train_data_file)
    num_data_points = len(train_data)
    if num_data_points < 1:
        print('  Not enough data points for %s, skipping' % dname)
        return
    print('\t# Total data points for %s: %d ' % (dname, num_data_points))

    X_feature = train_data.drop(['device', 'state', 'event', 'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    y_labels = np.array(train_data.state)
    train_hosts = np.array(train_data.hosts)
    train_protocol = np.array(train_data.protocol)

    '''
    read idle data for training (as negatives)
    '''
    train_bg = pd.read_csv("data/idle-2021-train-std/%s.csv" % dname)
    bg_feature = train_bg.drop(['device', 'state', 'event', 'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    bg_feature = np.array(bg_feature)
    bg_labels = np.zeros(len(bg_feature))

    '''
    test for a specific activity
    '''
    # print()
    # X_feature = X_feature.loc[train_data['state'] == 'android_lan_off']
    # X_on = train_data.loc[train_data['state'] == 'android_lan_off']
    # X_on = X_on.drop(['device', 'state', 'event', 'hosts'], axis=1).fillna(-1)
    # mu = X_on.mean(axis=0).values
    # sigma = X_on.cov().values
    # model = multivariate_normal(cov=sigma, mean=mu, allow_singular=True)

    # X_off = train_data.loc[train_data['state'] == 'alexa_off']
    # X_off = X_off.drop(['device', 'state', 'event', 'hosts'], axis=1).fillna(-1)
    # pred_y1 = model.logpdf(X_on.values)
    # print(pred_y1)
    # pred_y = model.logpdf(X_off.values)
    # print(pred_y)
    # return []
    # print('Number of labels:', num_lables)
    # print('After iloc:',X_feature.shape)
    # print(X_feature.shape)



    """
    read testing set
    """
    test_data = pd.read_csv(os.path.join('data/test-filtered-std/', '%s.csv' %dname))

    X_test = test_data.drop(['device', 'state', 'event', 'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    y_test = np.array(test_data.state)
    test_hosts = np.array(test_data.hosts)
    test_protocol = np.array(test_data.protocol)
    test_events = np.array(test_data.event)
    
    idle_FP_data = pd.read_csv(os.path.join('data/idle-2021-test-filtered-std/', '%s.csv' %dname))
    idle_FP_test = idle_FP_data.drop(['device', 'state', 'event', 'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    idle_FP_ts = np.array(idle_FP_data.start_time)
    idle_FP_protocol = np.array(idle_FP_data.protocol)
    idle_FP_host = np.array(idle_FP_data.hosts)

    # aggregate indistinguishable labels
    y_labels, y_test = utils.label_aggregate(y_labels, y_test, dname)

    X_train = X_feature
    y_train = y_labels


    print('Train: %s' % len(X_train))
    print('features:', np.shape(X_train))
    print('Test: %s' % len(X_test))
    if len(X_test) == 0:
        print('Not enough testing sample')
        return

    num_lables = len(set(y_train))


    # read fingerprints domains
    fingerprint_file = 'model/fingerprint/%s.txt' % dname
    activity_fingerprint_dic = {}
    activity_fingerprint_merge_count = {}
    tmp_activity_list = []
    tmp_hostname_list = []
    with open(fingerprint_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line.startswith('fingerprint'):
                tmp_activity_list.append(line.split(':')[0].split('- ')[1])
                tmp_hostname_list.append(line.split(': ')[1].split(';')
                )
    tmp_activity_list, _ = utils.label_aggregate(tmp_activity_list, tmp_activity_list, dname)
    for i in range(len(tmp_activity_list)):
        tmp_activity = tmp_activity_list[i]
        tmp_host = tmp_hostname_list[i]
        
        if tmp_activity not in activity_fingerprint_dic:
            activity_fingerprint_dic[tmp_activity] = tmp_host
            activity_fingerprint_merge_count[tmp_activity] = 1
        else:
            for x in tmp_host:
                activity_fingerprint_dic[tmp_activity].append(x)
            activity_fingerprint_merge_count[tmp_activity] += 1
                # print(activity_fingerprint_merge_count)

    for activity in activity_fingerprint_dic.keys():
        tmp_activity_list = []
        # print(activity, set(activity_fingerprint_dic[activity]))
        for i in set(activity_fingerprint_dic[activity]):
            if activity_fingerprint_merge_count[activity] == activity_fingerprint_dic[activity].count(i):
                tmp_activity_list.append(i)
        activity_fingerprint_dic[activity] = tmp_activity_list

    print(dname, activity_fingerprint_dic, activity_fingerprint_merge_count)
    # return 0 
    # return 
    """
    Labels encoding 
    """
    lb = LabelBinarizer()
    lb.fit(y_train)  

    # 
    # set of all labels
    positive_label_set = lb.classes_.tolist()
    # list for predicted results
    predict_labels_agg = []
    for i in range(len(y_test)):
        predict_labels_agg.append([])
    
    """
    Train through binary ML algorithms
    """

    # model files, label files, and output files
    model_dir = os.path.join(root_model, model_alg)
    if not os.path.exists(model_dir):
        os.system('mkdir -pv %s' % model_dir)
    
    label_file = os.path.join(model_dir, dname + ".label.txt")
    output_file = os.path.join(root_output, "result_" + model_alg + ".txt")
    with open(output_file,'a+') as of:
        of.write('---%s---\n' % dname)

    print('Label set:', positive_label_set)
    idle_fp_set = set() # Idel FP = 

    train_host_protocol = ["%s,%s"%(i,j) for i, j in zip(train_hosts,train_protocol)]
    if len(train_host_protocol) != len(y_train):
        print(dname,  len(train_host_protocol) , len(y_train))
        exit(1)
    test_host_protocol = ["%s,%s"%(i,j) for i, j in zip(test_hosts,test_protocol)]
    idle_host_protocol = ["%s,%s"%(i,j) for i, j in zip(idle_FP_host,idle_FP_protocol)]
    '''
    Start for-loop of all positive label set
    '''
    for positive_label in positive_label_set:
        """
        Set new binary labels for training and testing set
        """
        if positive_label not in activity_fingerprint_dic:
            print('Error: positive label not in activity_fingerprint_dic')
            exit(1)
        
        y_train_tmp = np.array(y_train)
        y_test_tmp = np.array(y_test)

        filter_tmp = []
        filter_test_tmp = []
        for i in range(len(y_train_tmp)):
            if y_train_tmp[i] != positive_label:
                filter_tmp.append(False)
                y_train_tmp[i] = 0
            else:
                y_train_tmp[i] = 1
                filter_tmp.append(True)
        for i in range(len(y_test_tmp)):
            if y_test_tmp[i] != positive_label:
                filter_test_tmp.append(False)
                y_test_tmp[i] = 0
            else:
                filter_test_tmp.append(True)
                y_test_tmp[i] = 1

        # Current positive label and the number of the positive label 
        print('------%s cur label len: %d-----' % (positive_label, len(y_train_tmp[filter_tmp])))


        # Event: one experiment (may contains multiple flows)
        test_events_positive = test_events[filter_test_tmp]
        print('Testing-set positive event set len:',len(set(test_events_positive)))
        num_of_events = len(set(test_events_positive))

        cur_fingerprint_list = activity_fingerprint_dic[positive_label] # set of domain_name,protocol
        for cur_host_protocol_tup in cur_fingerprint_list:
            if cur_host_protocol_tup == '':
                continue
            
            filter_host_protocol = []
            for i in range(len(train_host_protocol)):
                if cur_host_protocol_tup.split(',')[0].startswith('*'):
                    matched_suffix = train_hosts[i].endswith(cur_host_protocol_tup.split(',')[0][2:])
                else:
                    matched_suffix = False
                
                if y_train_tmp[i]== positive_label and train_host_protocol[i] != cur_host_protocol_tup and matched_suffix == False:
                    filter_host_protocol.append(False)
                else:
                    filter_host_protocol.append(True)
            cur_X_train_tmp = X_train[filter_host_protocol]
            cur_y_train_bin = y_train_tmp[filter_host_protocol].tolist()
            if len(bg_labels) >  len(cur_X_train_tmp):
                bg_undersampling = len(cur_X_train_tmp)
            else:
                bg_undersampling = len(bg_labels)
            # bg_undersampling = round(len(bg_labels)*0.005)
            bg_undersampling = 30 if bg_undersampling < 100 else bg_undersampling
            random.shuffle(bg_feature)

            # Concatenate training set with idle set. 1 stands for positive label, 0 stands for other activity labels and idle traffic
            cur_y_train_bin = np.concatenate((cur_y_train_bin,bg_labels[:bg_undersampling]),axis=0)
            cur_X_train_tmp = np.concatenate((cur_X_train_tmp,bg_feature[:bg_undersampling]),axis=0)
            # cur_train_host_protocol = np.concatenate((train_host_protocol,bg_labels[:bg_undersampling]),axis=0)

            cur_y_train_bin = cur_y_train_bin.tolist()
            y_test_bin_1d = y_test_tmp.tolist()
            """
            Training
            """
                
            print(model_dir)
            if not os.path.exists(model_dir + '/' + dname ):
                os.mkdir(model_dir + '/' + dname )
            model_file = os.path.join(model_dir + '/' + dname + '/' +
                model_alg + '_' + positive_label + '_' + cur_host_protocol_tup + ".model")
            trained_model = RandomForestClassifier(n_estimators=100, random_state=100)
            trained_model.fit(cur_X_train_tmp, cur_y_train_bin)
            pickle.dump(trained_model, open(model_file, 'wb'))

            """
            Testing
            """

            y_predicted = trained_model.predict(X_test) 
            y_proba = trained_model.predict_proba(X_test)
            if len(idle_FP_test)!=0:
                idle_predicted = trained_model.predict(idle_FP_test)
            else:
                idle_predicted = []

            # calculate TP, FP, FN, FP
            tp_events = set()
            fp_events = set()
            fn_events = set()
            count_p = 0
            for i in range(len(y_predicted)):
                if cur_host_protocol_tup.split(',')[0].startswith('*'):
                    matched_suffix = test_hosts[i].endswith(cur_host_protocol_tup.split(',')[0][2:])
                else:
                    matched_suffix = False
                 
                if y_test_bin_1d[i] == 1 and (test_host_protocol[i] == cur_host_protocol_tup or (test_protocol[i]==cur_host_protocol_tup.split(',')[1] and matched_suffix)):
                    count_p += 1
                    if y_predicted[i] == 1: # TP: as long as one flow in an event is correctly classified as positive, count it as TP
                        tp_events.add(test_events[i])
                    else: # FN
                        pass
                    predict_labels_agg[i].append((positive_label, y_proba[i][1]))
            print(dname, cur_host_protocol_tup, num_of_events, count_p)

            # Flow level metrics: 
            precision, recall, f1, support = precision_recall_fscore_support(y_test_bin_1d, y_predicted,average='binary') # , average='micro'
            # idle FP test
            for i in range(len(idle_predicted)):
                if cur_host_protocol_tup.split(',')[0].startswith('*'):
                    matched_suffix = idle_FP_host[i].endswith(cur_host_protocol_tup.split(',')[0][2:])
                else:
                    matched_suffix = False
                 
                if idle_predicted[i] == 1 and (idle_host_protocol[i] == cur_host_protocol_tup or (matched_suffix and idle_FP_protocol[i]==cur_host_protocol_tup.split(',')[1])):
                    idle_fp_set.add(idle_FP_ts[i]) # For idle, all y_test_bin are 0, all predicted 1 are FPs
            if num_of_events == 0:
                continue
            with open(output_file,'a+') as of:
                # of.write('%s, acc: %.2f \n' % (positive_label, count_fp/len(y_predicted_1d))) # for idle
                of.write('%s, %s, flow-level TP: %d, event-level recall: %.2f, F1: %.2f\n' % (
                    positive_label, cur_host_protocol_tup, len(tp_events), len(tp_events)/num_of_events, f1))
    
    '''
    End for-loop 
    '''
    num_of_events = len(set(test_events))
    
    '''
    Step 1: combine resutls from All binary models. Sort results by probability
    Return: For each flow: the list ordered by probability
    '''
    for i in range(len(predict_labels_agg)):
        if len(predict_labels_agg[i]) == 0:
            predict_labels_agg[i] = [('000',0)]
        if len(predict_labels_agg[i]) > 1:  # if match more than one binary models. 
            tmp_list = predict_labels_agg[i]
            tmp_list = sorted(tmp_list,key=lambda t: t[-1],reverse=True) # sort by proba
            predict_labels_agg[i] = tmp_list


    '''
    Step 2: combine results within a time window by majority voting and proba
    '''
    #### Activity testing set: 
    event_dict = {} # predicted and actual labels comparison. 
    event_dict_log = {} # For each flow, pick the one with largest proba
    for i in range(len(predict_labels_agg)):
        cur_label = predict_labels_agg[i][0][0]
        cur_proba = predict_labels_agg[i][0][1]
        cur_host_protocol = test_host_protocol[i]
        if test_events[i] in event_dict:
            event_dict_log[test_events[i]].append((cur_label, cur_proba,  cur_host_protocol))
            # if cur_proba > event_dict[test_events[i]][0][1]:
            #     event_dict[test_events[i]] = [(cur_label, cur_proba), y_test[i]]
        else:
            event_dict[test_events[i]] = [0, y_test[i]]
            event_dict_log[test_events[i]] = [(cur_label, cur_proba,  cur_host_protocol)]

    # print(len(event_dict_log.keys()), event_dict_log)
    for k, v in event_dict_log.items():
        # v: list of (label, proba, timestamp)
        tmp_label_list = []
        tmp_proba_list = []
        tmp_host_protocol_list = []
        for i in v:
            tmp_label_list.append(i[0])
            tmp_proba_list.append(i[1])
            tmp_host_protocol_list.append(i[2])

        counter = 0
        predicted_label = tmp_label_list[0] # final label
        proba = 0
        for i in range(len(tmp_label_list)):    # all candidate labels
            curr_frequency = tmp_label_list.count(tmp_label_list[i])
            if curr_frequency > counter and tmp_label_list[i]!='000':
                counter = curr_frequency
                predicted_label = tmp_label_list[i]
                proba = tmp_proba_list[i]
            elif(curr_frequency == counter and tmp_label_list[i]!='000'):
                if(tmp_proba_list[i] > proba):
                    predicted_label = tmp_label_list[i]
                    proba = tmp_proba_list[i]
        
        if predicted_label == '000':
            event_dict[k] = [predicted_label, event_dict[k][-1]]    # 
            continue
        cur_fingerprint_list = activity_fingerprint_dic[predicted_label]
        all_fingerprint_matched = 1
        for cur_tup in cur_fingerprint_list:
            if cur_tup not in tmp_host_protocol_list:
                all_fingerprint_matched = 0
        if all_fingerprint_matched:
            event_dict[k] = [predicted_label, event_dict[k][-1]]
        else:
            event_dict[k] = ['000', event_dict[k][-1]]
        # cur_label = max(set(tmp_list), key = tmp_list.count)
        event_dict[k] = [predicted_label, event_dict[k][-1]]    # predicted label, true label
    

    '''
    Step 3: calculate event level results
    '''
    new_true_label = []
    new_predict_label = []
    # for k,v in event_dict.items():
    # print(event_dict)
    for k,v in event_dict.items():
        if v[1]!=v[0]:
            print(k,v)
        new_true_label.append(v[1])
        new_predict_label.append(v[0])
    
    events_num = len(new_true_label)
    try:
        lb = LabelEncoder()
        lb.fit(new_predict_label+new_true_label)
        new_true_label_1d = lb.transform(new_true_label)
        new_predict_label_1d = lb.transform(new_predict_label)
        tp_num = 0
        for i in range(len(new_true_label_1d)):
            if new_true_label_1d[i]==new_predict_label_1d[i]:
                tp_num += 1
        _, _, event_f1, _ = precision_recall_fscore_support(new_true_label_1d, new_predict_label_1d, average='micro')
        event_precision, event_recall, _, _ = precision_recall_fscore_support(new_true_label_1d, new_predict_label_1d, average='macro')
    except:
        event_precision = -1
        event_recall = -1
        event_f1 = -1
    
    if events_num != 0:
        try:
            with open(output_file,'a+') as of:
                of.write('\n%s, number of events: %d, classified correct: %d, event macro recall: %.2f, event macro precision: %.2f, event micro F1: %.2f\n' % (
                    'Average', events_num, tp_num, event_recall, event_precision, event_f1))
        except:
            pass

    """
    Idle evaluation:
    comment out if unnescessary
    """
    with open(output_file,'a+') as of:
        if len(idle_FP_test)!=0:
            of.write('Idle traffic: Overall FP rate: %.2f, # of FP: %d , %d\n' % (len(idle_fp_set)/len(idle_FP_test), len(idle_fp_set), len(idle_FP_test))) # for idle
        else:
            of.write('Idle traffic: Overall FP rate: 0, # of FP: 0 , 0\n' )

    '''
    Logs
    '''
    print('# event_dict length',len(event_dict.keys()))
    print('------------------------------------------------')
    if not os.path.exists('%s/logs/' %  root_model):
        os.mkdir('%s/logs/' %  root_model)
    with open('%s/logs/%s-test.txt' % (root_model, dname), 'w+') as off:
        for k, v in event_dict.items():
            off.write("%s %s :%s\n" % (k,v[-1],v[0]))

    return []
        
if __name__ == '__main__':
    main()
    num_pools = 1
