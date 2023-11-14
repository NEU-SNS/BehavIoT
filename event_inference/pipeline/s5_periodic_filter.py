import warnings
import utils
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import DBSCAN
import time

import Constants as c


warnings.simplefilter("ignore", category=DeprecationWarning)


num_pools = 1

cols_feat = utils.get_features()


model_list = []
root_output = ''
dir_tsne_plots = ''
root_feature = ''
root_model = ''
dbscan_eps = 1
#is_error is either 0 or 1
def print_usage(is_error):
    print(c.PERIODIC_MOD_USAGE, file=sys.stderr) if is_error else print(c.PERIODIC_MOD_USAGE)
    exit(is_error)

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
    # Result is noise by default   euclidean_distances
    # pass
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):

            if metric(x_new, x_core) < (dbscan_model.eps): # np.reshape(x_new, (1,-1)), np.reshape(x_core,(1,-1))
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

def main():
    # test()
    global  root_output, dir_tsne_plots, model_list , root_feature, root_model, dbscan_eps

    # Parse Arguments
    parser = argparse.ArgumentParser(usage=c.PERIODIC_MOD_USAGE, add_help=False)
    parser.add_argument("-i", dest="root_feature", default="")
    parser.add_argument("-o", dest="root_model", default="")
    parser.add_argument("-e", dest="eps", default=1)

    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    if args.help:
        print_usage(0)

    print("Running %s..." % c.PATH)

    # Error checking command line args
    root_feature = args.root_feature
    root_model = args.root_model
    dbscan_eps = args.eps
    errors = False
    #check -i in features
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

    #check -o out models
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


    if errors:
        print_usage(1)
    #end error checking

    print("Input files located in: %s\nOutput files placed in: %s" % (root_feature, root_model))
    train_models()


def train_models():
    global root_feature, root_model, root_output, dir_tsne_plots
    """
    Scan feature folder for each device
    """
    print('root_feature: %s' % root_feature)
    print('root_model: %s' % root_model)
    print('root_output: %s' % root_output)
    lfiles = []
    lparas = []
    ldnames = []
    # for i in range(5):
    # random_state = random.randint(0, 1000)
    random_state = 422
    print("random_state:", random_state)
    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            print(csv_file)
            train_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]

            lparas.append((train_data_file, dname, random_state))
    # p = Pool(num_pools)
    t0 = time.time()
    # exit(1)
    for i in range(len(lparas)):
        list_results = eid_wrapper(lparas[i])
    # list_results = p.map(eid_wrapper, lparas)
    # for paras in lparas:
    #     list_results = eid_wrapper(paras)
    # print(list_results)
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
    return eval_individual_device(a[0], a[1], a[2])


def eval_individual_device(train_data_file, dname, random_state, specified_models=None):
    global root_feature, root_model, root_output, dbscan_eps
    """
    Assumptions: the train_data_file contains only 1 device, all possible labels
    """

    """
    Prepare the directories and add only models that have not been trained yet 
    """
    model_alg = 'filter'
    model_dir = '%s/%s' % (root_model, model_alg)
    model_file = '%s/%s%s.model' % (model_dir, dname, model_alg)

    """
    Training file reading 
    """

    print('Training %s ' % (dname))
    train_data = pd.read_csv(train_data_file)

    num_data_points = len(train_data)
    if num_data_points < 10:
        print('  Not enough data points for %s' % dname)
        # return
    print('\t#Total data points: %d ' % num_data_points)


    """
    Get period from fingerprinting files
    """
    periodic_tuple = []
    tmp_host_set = set()
    try:
        with open('./period_detection/freq_period/2021_fingerprints/%s.txt' % dname, 'r') as file:
            for line in file:
                tmp = line.split()

                try:
                    tmp_proto = tmp[0]
                    tmp_host = tmp[1]
                    tmp_period = tmp[2]
                except:
                    print(tmp)
                    exit(1)
                if tmp_host == '#' or tmp_host  == ' ':
                    tmp_host = ''

                periodic_tuple.append((tmp_host, tmp_proto, tmp_period))
                tmp_host_set.add((tmp_host,tmp_proto))

    except:
        print( 'unable to read fingerprint file')
        return
    print(dname, periodic_tuple)
    
    """
    Preprocess training data
    """

    X_feature = train_data.drop(['device', 'state', 'event','start_time', 'protocol', 'hosts'], axis=1).fillna(-1)

    protocols = train_data['protocol'].fillna('').values
    hosts = train_data['hosts'].fillna('').values
    protocols = utils.protocol_transform(protocols)
    
    for i in range(len(hosts)):
        if hosts[i] != '' and hosts[i] != None:
            try:
                tmp = hosts[i].split(';')
            except:
                print(hosts[i])
                exit(1)
            hosts[i] = tmp[0]
        if hosts[i] == None:
            hosts[i] == 'non'
        hosts[i] = hosts[i].lower()
            # print(hosts[i])

    # y_labels = np.array(train_data.state)
    X_feature = np.array(X_feature)
    print('X_feature.shape:',X_feature.shape)
    


    """
    Load and preprocess testing data
    """
    print('loading test data')

    test_data = pd.read_csv("data/idle-2021-test-std/%s.csv" % dname)
    test_feature = test_data.drop(['device', 'state', 'event', 'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    test_data_numpy = np.array(test_data)
    test_feature = np.array(test_feature)
    test_protocols = test_data['protocol'].fillna('').values
    test_hosts = test_data['hosts'].fillna('').values
    test_protocols = utils.protocol_transform(test_protocols)

    
    for i in range(len(test_hosts)):
        if test_hosts[i] != '' and test_hosts[i] != None:
            try:
                tmp = test_hosts[i].split(';')
            except:
                print(test_hosts[i])
                exit(1)
            test_hosts[i] = tmp[0]
        if test_hosts[i] == None:
            test_hosts[i] == 'non'
        test_hosts[i] = test_hosts[i].lower()

    events = test_data['event'].fillna('').values
    len_test_before = len(test_feature)
    num_of_event = len(set(events))
    y_labels_test = test_data['state'].fillna('').values


    """
    # filter out local and DNS/NTP packets
    """
    print('testing data len: ', len_test_before)
    filter_dns = []
    for i in range(len(test_feature)):
        if (test_hosts[i] == 'multicast' or test_protocols[i] == 'DNS' or test_protocols[i] == 'MDNS' 
            or test_protocols[i] == 'NTP' or test_protocols[i] == 'SSDP' or test_protocols[i] == 'DHCP'):
            filter_dns.append(False)
        else:
            filter_dns.append(True)
    test_feature = test_feature[filter_dns]
    test_hosts = test_hosts[filter_dns]
    test_protocols = test_protocols[filter_dns]
    events = events[filter_dns]
    y_labels_test = y_labels_test[filter_dns]
    test_data_numpy = test_data_numpy[filter_dns]

    print('testing data after DNS/NTP etc filter: ', len(test_feature))
    
    ret_results = []
    res_left = 0
    res_filtered = 0
    ## For each tuple: 
    for tup in periodic_tuple:
        tmp_host = tup[0]
        tmp_proto = tup[1]

        print('------%s------' %dname)
        print(tmp_proto, tmp_host)
        

        filter_l = []
        for i in range(len(X_feature)):
            if tmp_host.startswith('*'):
                matched_suffix = hosts[i].endswith(tmp_host[2:])
            else:
                matched_suffix = False
            if (hosts[i] == tmp_host or matched_suffix) and protocols[i] == tmp_proto:
                filter_l.append(True)
            else:
                filter_l.append(False)
        X_feature_part = X_feature[filter_l]
        print('train feature part:',len(X_feature_part))
        x_zero_feature_flag = 0
        if len(X_feature_part) == 0:
            x_zero_feature_flag = 1
        if len(X_feature_part) > 5000:
            X_feature_part = X_feature_part[:5000]
        """
        ML algorithms
        """

        if not os.path.exists(model_dir):
            os.system('mkdir -pv %s' % model_dir)
        model_file = os.path.join(model_dir, dname + tmp_host + tmp_proto +".model")

        """
        Two steps
            1. Train 
            2. Test 
            3. Evaluate 
        """
        X_feature_part = pd.DataFrame(X_feature_part)
        if len(X_feature_part) == 0:
            print('Not enough training data for %s' % tmp_host)
            continue
        print("predicting by trained_model")
        print('Test len before:',len(test_feature))
        filter_test = []
        matched_suffix = False
        for i in range(len(test_feature)):
            
            if tmp_host.startswith('*'):
                matched_suffix = test_hosts[i].endswith(tmp_host[2:])
                if matched_suffix==False and tmp_host=='*.compute.amazonaws.com':
                    if test_hosts[i].endswith('.compute-1.amazonaws.com'):
                        matched_suffix =True
            else:
                matched_suffix = False
            if (test_hosts[i] == tmp_host or matched_suffix) and test_protocols[i] == tmp_proto:
                filter_test.append(True)    # for current (host + protocol)
            else:
                filter_test.append(False)
        test_feature_part = test_feature[filter_test]

        if len(test_feature_part) == 0:
            filter_test = []
            for i in range(len(test_feature)):
                if (test_hosts[i].endswith('.'.join(tmp_host.split('.')[-3:]))) and test_protocols[i] == tmp_proto:
                    filter_test.append(True)    # for current (host + protocol)
                else:
                    filter_test.append(False)

        events_part = events[filter_test]
        y_labels_test_part = y_labels_test[filter_test]
        ## DBSCAN
        eps_list = {}
        if dname in eps_list : 
            eps = eps_list[dname]
        else:
            eps = 5
        model = DBSCAN(eps=eps,min_samples=5)

        if x_zero_feature_flag == 0:
            y_train = model.fit_predict(X_feature_part)
        else: 
            y_train = model.fit_predict(test_feature_part)

        
        #TODO 
        if len(test_feature_part) == 0:
            print('test feature matched host/proto == 0') 
            model_dictionary = dict({'trained_model':model})
            pickle.dump(model_dictionary, open(model_file, 'wb'))
            model = 0
            continue
        print(test_feature_part.shape)


        y_new = dbscan_predict(model,test_feature_part)

        count_left = 0
        event_after = set()
        events_tmp = set()
        state_after = set()
        filter_list = []

        print('Training set average prediction: ',len(y_train), np.mean(y_train),np.var(y_train), np.mean(y_train) - 2 * np.var(y_train), np.count_nonzero(y_train==-1)) # np.count_nonzero(y_train==-1))#)
        print('testing set average prediction: ',len(y_new), np.mean(y_new), np.var(y_new), np.mean(y_new) - 2 * np.var(y_new), np.count_nonzero(y_train==-1) ) #np.count_nonzero(y_new==-1))

        for i in range(len(y_new)):
            if y_new[i] < 0: 
                event_after.add(events_part[i])
                state_after.add(y_labels_test_part[i])
                count_left += 1
                filter_list.append(True)

            else:
                filter_list.append(False)   # periodic traffic
        # activity_feature = test_feature_part[filter_list]
        if len(filter_list) != len(y_new):
            exit(1)
        count_tmp = 0
        for i in range(len(filter_test)):
            if filter_test[i] == False:
                filter_test[i] = True

            elif filter_test[i] == True: # true, (proto, host)
                if filter_list[count_tmp] == False: # filter
                    filter_test[i] = False
                count_tmp += 1
            else:
                exit(1)
        
        if len(filter_test) != len(test_feature):
            exit(1)


        test_feature = test_feature[filter_test]
        test_hosts = test_hosts[filter_test]
        test_protocols = test_protocols[filter_test]
        events = events[filter_test]
        y_labels_test = y_labels_test[filter_test]
        test_data_numpy = test_data_numpy[filter_test]
        
        res_left += count_left
        res_filtered += test_feature_part.shape[0] - count_left
        print("count_left" , count_left, test_feature_part.shape[0] , count_left/test_feature_part.shape[0])
        print('Test len after:',len(test_feature))
        print('-------------')

        """
        Save the model / logs
        """
        log_dir = os.path.join(root_model, 'logs')
        os.system('mkdir -pv %s' % log_dir)
        with open(os.path.join(log_dir,'%s.txt' % dname),'a+') as f:
            f.write('%s %s: ' % (tmp_proto, tmp_host))
            f.write('\nFlows left: %d %d %2f\n\n' % (count_left, test_feature_part.shape[0], count_left/test_feature_part.shape[0] ))

        model_dictionary = dict({'trained_model':model})
        pickle.dump(model_dictionary, open(model_file, 'wb'))
        model = 0

    
    """
    Save idle testing set
    """
    print('Flows left: ', len(test_feature)/len_test_before,len(test_feature), len(test_hosts), len_test_before)
    print('Activity left: ',len(set(test_data_numpy[:,-4]))/num_of_event, len(set(test_data_numpy[:,-4])), num_of_event)
    from collections import Counter
    print(Counter(test_hosts))
    with open(os.path.join(root_model,'results.txt'),'a+') as f:
        f.write('%s' % dname)
        f.write('\nFlows left: %2f %d %d' % (len(test_feature)/len_test_before,len(test_feature), len_test_before))
        f.write('\nActivity left: %2f %d %d \n\n' % (len(set(test_data_numpy[:,-4]))/num_of_event, len(set(test_data_numpy[:,-4])), num_of_event))

    test_feature = pd.DataFrame(test_data_numpy, columns=cols_feat) # use this only when processing std data 
    idle_filter_dir = 'data/idle-2021-test-filtered-std'
    if not os.path.exists(idle_filter_dir):
        os.mkdir(idle_filter_dir)
    filtered_train_processed= '%s/%s.csv' % (idle_filter_dir, dname)
    test_feature.to_csv(filtered_train_processed, index=False)
    return 0



if __name__ == '__main__':
    main()
    num_pools = 1

# 