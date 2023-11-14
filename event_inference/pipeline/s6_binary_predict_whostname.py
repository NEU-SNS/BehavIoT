import warnings
import os
import sys
import utils
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
from multiprocessing import Pool
import Constants as c

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

num_pools = 1


default_models = ['rf']
model_list = ['rf']
root_output = ''
root_feature = ''
root_model = ''
mac_dic = {}


short_window_device = ['tplink-plug', 't-wemo-plug', 'amazon-plug', 'tplink-bulb', 'smartlife-bulb',
'bulb1', 'magichome-strip', 'gosund-bulb1', 'govee-led1', 'meross-dooropener', 'nest-tstat', 'switchbot-hub']
long_window_device = ['wyze-cam','ikettle', 'echospot', 'dlink-camera', 'ring-camera', 'ring-doorbell']
time_window_dic = {}

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
            input_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]
            lparas.append((input_data_file, dname, random_state))
    p = Pool(num_pools)
    t0 = time.time()
    # list_results = p.map(eid_wrapper, lparas)
    for e in lparas:
        eid_wrapper(e)
    # for ret in list_results:
    #     if ret is None or len(ret) == 0: continue
    #     for res in ret:
    #         tmp_outfile = res[0]
    #         tmp_res = res[1:]
    #         with open(tmp_outfile, 'a+') as off:
    #             # off.write('random_state:',random_state)
    #             off.write('%s\n' % '\t'.join(map(str, tmp_res)))
    #             print('Agg saved to %s' % tmp_outfile)
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


def eval_individual_device(input_data_file, dname, random_state):
    # global root_feature, root_model, root_output

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
    read testing set
    """
    test_data = pd.read_csv(input_data_file)
    X_test = test_data.drop(['device', 'state', 'event', 'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    test_data_numpy = np.array(test_data)
    test_hosts = np.array(test_data.hosts)
    test_protocol = np.array(test_data.protocol)
    
    test_timestamp = np.array(test_data.start_time)

    test_protocol = utils.protocol_transform(test_protocol)

    if dname.startswith('tplink'):
        for i in range(len(test_hosts)):
            if test_hosts[i] == 'n-devs.tplinkcloud.com':
                test_hosts[i] = 'devs.tplinkcloud.com'

    print('Test: %s' % len(X_test))
    if len(X_test) == 0:
        print('Not enough testing sample')
        return

    # read fingerprints domains
    fingerprint_file = 'model/fingerprint/%s.txt' % dname
    if not os.path.exists(fingerprint_file):
        return
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
            # print(i, activity_fingerprint_merge_count[activity], activity_fingerprint_dic[activity].count(i))
            if activity_fingerprint_merge_count[activity] == activity_fingerprint_dic[activity].count(i):
                tmp_activity_list.append(i)
        activity_fingerprint_dic[activity] = tmp_activity_list

    print(dname, activity_fingerprint_dic, activity_fingerprint_merge_count)

    model_dir = os.path.join(root_model, model_alg)
    if not os.path.exists(model_dir):
        os.system('mkdir -pv %s' % model_dir)

    """
    Predict
    """
    positive_label_set = []
    
    if not os.path.exists(os.path.join(model_dir, dname)):
            return 0
    for f1 in os.listdir(os.path.join(model_dir, dname)):
        # print
        positive_label_set.append('_'.join(f1.split('_')[1:-1]))
    
    positive_label_set = set(positive_label_set)
    print(positive_label_set)

    predict_labels_agg = []
    X_test = np.array(X_test)
    test_host_protocol = ["%s,%s"%(i,j) for i, j in zip(test_hosts,test_protocol)]
    
    for i in range(len(X_test)):
        """
        Deviations, commment out if unnecessary 
        """
        predict_labels_agg.append([])
    for positive_label in positive_label_set:

        cur_fingerprint_list = activity_fingerprint_dic[positive_label]

        for cur_host_protocol_tup in cur_fingerprint_list:
            if cur_host_protocol_tup == '':
                continue

            model_file = os.path.join(model_dir , dname ,
                    model_alg + '_' + positive_label + '_' + cur_host_protocol_tup + ".model")
            trained_model = pickle.load(open(model_file, 'rb'))

            y_predicted = trained_model.predict(X_test) 
            y_proba = trained_model.predict_proba(X_test)

            if y_predicted.ndim == 1:
                y_predicted_1d = y_predicted
            else:
                y_predicted_1d = np.argmax(y_predicted, axis=1)

            
            count_cur = 0
            # try:
            for i in range(len(y_predicted_1d)):
                if cur_host_protocol_tup.split(',')[0].startswith('*'):
                    matched_suffix = test_hosts[i].endswith(cur_host_protocol_tup.split(',')[0][2:])
                else:
                    matched_suffix = False


                if (test_host_protocol[i] == cur_host_protocol_tup) or (test_protocol[i]==cur_host_protocol_tup.split(',')[1] and matched_suffix):
                    count_cur += 1
                    # print()
                    if y_predicted_1d[i] == 1:
                        predict_labels_agg[i].append((positive_label, y_proba[i][1]))

            print(cur_host_protocol_tup, count_cur, len(y_predicted_1d), len(predict_labels_agg))
    '''
    Evaluation
    '''
    '''
    Step 1: combine resutls from All binary models. Sort results by probability
    Return: predict_labels_agg: For each traffic flow, the list of predicted labels ordered by probability
    '''
    for i in range(len(predict_labels_agg)):
        if len(predict_labels_agg[i]) == 0:
            predict_labels_agg[i] = [('unknown',0)]
        elif y_predicted_1d[i] == -1:
            continue
        elif len(predict_labels_agg[i]) > 1:  # if match more than one binary models. 
            tmp_list = predict_labels_agg[i] # [(a,p_a), (b,p_b), ...]
            tmp_list = sorted(tmp_list,key=lambda t: t[-1],reverse=True) # sort by proba
            predict_labels_agg[i] = tmp_list
    
    '''
    Step 2: combine results within a time window by majority voting and proba
    TODO: time window
    Input: For each traffic flow, the list of predicted labels ordered by probability
    Output: label for each time window (aggregate labels from all flows within the window). 
    '''

    ## Sort by timestamp
    sorted_zip = sorted(zip(test_timestamp, predict_labels_agg, test_host_protocol, X_test, test_data_numpy))
    predict_labels_agg = [x for y, x, z , _, _ in sorted_zip]
    test_timestamp = [y for y, x, z, _, _ in sorted_zip]
    test_host_protocol = [z for y, x, z, _, _ in sorted_zip]
    X_test = np.array([a for y, x, z, a, b in sorted_zip])
    test_data_numpy = np.array([b for y, x, z, a, b in sorted_zip])

    #### 
    time_window_id = 0
    timestamp_bound = 0
    output_label_dic = {} # key: time window id, value: list of [(label, proba)]
    time_window_id_list = [] # the list associates time window id with traffic flow
    if dname in time_window_dic.keys():
        time_window_length = time_window_dic[dname]
    elif dname in long_window_device:
        time_window_length = 35
    else:
        # print(dname)
        # exit(1)
        time_window_length = 5
    
    for i in range(len(predict_labels_agg)):
        cur_label = predict_labels_agg[i][0][0]
        cur_proba = predict_labels_agg[i][0][1]
        cur_host_protocol = test_host_protocol[i]
        if len(predict_labels_agg[i]) > 1 and  predict_labels_agg[i][0][1] == predict_labels_agg[i][1][1]:
            print('Equal proba: ', predict_labels_agg[i])
        if test_timestamp[i] == -1 or test_timestamp[i] == 'NaN' or test_timestamp[i] == None:
            continue
        if test_timestamp[i] <= timestamp_bound:
            time_window_id_list.append(time_window_id)
            output_label_dic[time_window_id].append((cur_label,cur_proba, cur_host_protocol))
        else:
            time_window_id += 1
            time_window_id_list.append(time_window_id)
            timestamp_bound =  int(test_timestamp[i]) + time_window_length
            output_label_dic[time_window_id] = [(cur_label,cur_proba, cur_host_protocol)]
    if len(time_window_id_list) != len(predict_labels_agg):
        print("Error: time_window_id_list length is not consistent")
        exit(1)
    
    # aggregate labels by majority votes 

    for k,v in output_label_dic.items(): # for each time window id 
        # print(k,v)
        tmp_label_list = []
        tmp_proba_list = []
        tmp_host_protocol_list = []
        for i in v:
            tmp_label_list.append(i[0])
            tmp_proba_list.append(i[1])
            tmp_host_protocol_list.append(i[2])

        counter = 0
        cur_label = tmp_label_list[0] # final label
        proba = 0
        for i in range(len(tmp_label_list)):    # for each candidate label
            curr_frequency = tmp_label_list.count(tmp_label_list[i])
            if curr_frequency > counter and tmp_label_list[i]!='unknown':
                counter = curr_frequency
                cur_label = tmp_label_list[i]
                proba = tmp_proba_list[i]
            elif(curr_frequency == counter and tmp_label_list[i]!='unknown'):
                if(tmp_proba_list[i] > proba):
                    cur_label = tmp_label_list[i]
                    proba = tmp_proba_list[i]

        # final label cur_label
        all_fingerprint_matched = 1
        if cur_label == 'unknown':
            output_label_dic[k] = cur_label
            continue
        
        cur_fingerprint_list = activity_fingerprint_dic[cur_label]
        # print('cur_fingerprint_list ', cur_fingerprint_list)
        for cur_tup in cur_fingerprint_list:
            
            if cur_tup.split(',')[0].startswith('*'):
                not_matched = True
                for tmp_index in range(len(tmp_host_protocol_list)):
                    if tmp_host_protocol_list[tmp_index].split(',')[0].endswith(cur_tup.split(',')[0][2:]):
                        not_matched = False
                        
            else:
                not_matched = True
            if cur_tup not in tmp_host_protocol_list and not_matched:
                all_fingerprint_matched = 0
            
            
        if all_fingerprint_matched:
            output_label_dic[k] = cur_label
        else:
            output_label_dic[k] = 'unknown'

    output_label_list = []
    for i in range(len(time_window_id_list)):
        output_label_list.append(output_label_dic[time_window_id_list[i]])


    '''
    Step 3: calculate event level results
    Input:  output_label_list: list of final labels
            test_timestamp: timestamp list
            time_window_id_list: time window id list
    '''

    '''
    save unknown
    '''
    filter_list = []
    for i in range(len(time_window_id_list)):
        time_window_id = time_window_id_list[i]
        print(time_window_id, output_label_dic[time_window_id], test_timestamp[i])
        if output_label_dic[time_window_id] == 'unknown':
            filter_list.append(True)
        else:
            filter_list.append(False)
    X_test = X_test[filter_list]
    test_data_numpy_new = test_data_numpy[filter_list]
    test_feature = pd.DataFrame(X_test)
    test_feature['device'] = test_data_numpy_new[:,-6]
    test_feature['state'] = test_data_numpy_new[:,-5]
    test_feature['event'] = test_data_numpy_new[:,-4]
    test_feature['start_time'] = test_data_numpy_new[:,-3]
    test_feature['protocol'] = test_data_numpy_new[:,-2]
    test_feature['hosts'] = test_data_numpy_new[:,-1]

    output_dir = '%s-unknown/' % root_feature[:-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filtered_train_processed= '%s/%s.csv' % (output_dir , dname)
    # filtered_train_processed= 'data/test-filtered-std/%s.csv' % ( dname)
    print('Unknown csv:', filtered_train_processed)
    test_feature.to_csv(filtered_train_processed, index=False)
    '''
    Logs
    '''

    print('-----------------------logs-------------------------')

    dataset = root_feature.split('/')[1].split('-')[0]
    if not os.path.exists('%s/%s' % (root_model, dataset)):
        os.mkdir('%s/%s' % (root_model, dataset))
    with open('%s/%s/%s.txt' % (root_model, dataset, dname), 'w+') as off:
        for i in range(len(test_timestamp)):
            off.write("%s :%s, %s\n" % (datetime.fromtimestamp(test_timestamp[i]
                ).strftime("%m/%d/%Y, %H:%M:%S"), output_label_list[i], test_host_protocol[i]))

    output_log_file = 'logs/log_unknown_%s' % root_feature.split('/')[-2]

    if not os.path.exists(output_log_file):
        os.mkdir(output_log_file)
    with open('%s/test-%s.txt' % (output_log_file, dname), 'w+') as off:
        cur_time_window_id = 0
        for i in range(len(output_label_list)):
            if time_window_id_list[i] != cur_time_window_id:
                cur_time_window_id += 1
                off.write("%s :%s\n" % (datetime.fromtimestamp(test_timestamp[i]
                ).strftime("%m/%d/%Y, %H:%M:%S.%f"),output_label_list[i]))
            else:
                pass

    return



if __name__ == '__main__':
    main()
    num_pools = 1
