import warnings
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime
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
    print("random_state:", random_state)
    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            print(csv_file)
            input_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]
            lparas.append((input_data_file, dname, random_state))
    p = Pool(num_pools)
    t0 = time.time()
    list_results = p.map(eid_wrapper, lparas)
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
        input_data_file, dname, random_state
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
    test_timestamp = np.array(test_data['start_time'])

    print('Test: %s' % len(X_test))
    if len(X_test) == 0:
        print('Not enough testing sample')
        return

    """
    Trained model
    """
    # model files, label files, and output files
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
        positive_label_set.append('_'.join(f1.split('.')[0].split('_')[1:]))
    
    print(positive_label_set)

    y_total_predict = []
    predict_labels_agg = []
    X_test = np.array(X_test)

    
    for i in range(len(X_test)):
        """
        Deviations, commment out if unnecessary 
        """
        predict_labels_agg.append([])

    for positive_label in positive_label_set:
        model_file = os.path.join(model_dir, dname, model_alg + '_' + positive_label + ".model")
        trained_model = pickle.load(open(model_file, 'rb'))

        y_predicted = trained_model.predict(X_test) 
        y_proba = trained_model.predict_proba(X_test)
        if y_predicted.ndim == 1:
            y_predicted_1d = y_predicted
        else:
            y_predicted_1d = np.argmax(y_predicted, axis=1)

        for i in range(len(y_predicted_1d)):
            if y_predicted_1d[i] == 1:
                predict_labels_agg[i].append((positive_label, y_proba[i][1]))
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
    Input: For each traffic flow, the list of predicted labels ordered by probability
    Output: label for each time window (aggregate labels from all flows within the window). 
    '''
    ## Sort by timestamp
    sorted_zip = sorted(zip(test_timestamp, predict_labels_agg))
    predict_labels_agg = [x for y, x in sorted_zip]
    test_timestamp = [y for y, x in sorted_zip]

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
        # cur_host = test_hosts[i]
        # cur_protocol = test_protocol[i]
        if len(predict_labels_agg[i]) > 1 and  predict_labels_agg[i][0][1] == predict_labels_agg[i][1][1]:
            print('Equal proba: ', predict_labels_agg[i])
        if test_timestamp[i] == -1 or test_timestamp[i] == 'NaN' or np.isnan(test_timestamp[i]):
            print(test_timestamp[i])
            continue
        elif test_timestamp[i] <= timestamp_bound:
            time_window_id_list.append(time_window_id)
            output_label_dic[time_window_id].append((cur_label,cur_proba))
        else:
            time_window_id += 1
            time_window_id_list.append(time_window_id)
            timestamp_bound =  int(test_timestamp[i]) + time_window_length
            output_label_dic[time_window_id] = [(cur_label,cur_proba)]
    if len(time_window_id_list) != len(predict_labels_agg):
        print("Error: time_window_id_list length is not consistent")
        exit(1)
    
    # aggregate labels by majority votes 
    for k,v in output_label_dic.items(): # for each time window id 
        tmp_label_list = []
        tmp_proba_list = []
        for i in v:
            tmp_label_list.append(i[0])
            tmp_proba_list.append(i[1])

        counter = 0
        cur_label = tmp_label_list[0] # final label
        proba = 0
        for i in range(len(tmp_label_list)):    # for each candidate label
            curr_frequency = tmp_label_list.count(tmp_label_list[i])
            if curr_frequency > counter and tmp_label_list[i]!='000':
                counter = curr_frequency
                cur_label = tmp_label_list[i]
                proba = tmp_proba_list[i]
            elif(curr_frequency == counter and tmp_label_list[i]!='000'):
                if(tmp_proba_list[i] > proba):
                    cur_label = tmp_label_list[i]
                    proba = tmp_proba_list[i]
        
        output_label_dic[k] = cur_label


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
    Logs
    '''

    print('-----------------------logs-------------------------')


    if not os.path.exists('%s/unctrl' % (root_model)):
        os.mkdir('%s/unctrl' % (root_model))
    with open('%s/unctrl/%s.txt' % (root_model, dname), 'w+') as off:
        for i in range(len(test_timestamp)):
            off.write("%s :%s\n" % (datetime.fromtimestamp(test_timestamp[i]
                ).strftime("%m/%d/%Y, %H:%M:%S"), output_label_list[i]))

    
    output_log_file = 'logs/log_%s' % root_feature.split('/')[-2]

    if not os.path.exists(output_log_file):
        os.mkdir(output_log_file)
    with open('%s/unctrl-%s.txt' % (output_log_file, dname), 'w+') as off:
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
