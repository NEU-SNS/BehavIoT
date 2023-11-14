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

from multiprocessing import Pool
import Constants as c


warnings.simplefilter("ignore", category=DeprecationWarning)

num_pools = 12
cols_feat = utils.get_features()

model_list = []
root_output = ''
dir_tsne_plots = ''
dataset = ''
root_model = ''
dbscan_eps = 1
mac_dic = {}
#is_error is either 0 or 1
def print_usage(is_error):
    print(c.PERIODIC_MOD_USAGE, file=sys.stderr) if is_error else print(c.PERIODIC_MOD_USAGE)
    exit(is_error)

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
    # Result is noise by default   euclidean_distances
    # pass
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 


    # print('shape y_new:',len(y_new))
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        # print(dbscan_model.components_)
        for i, x_core in enumerate(dbscan_model.components_):
            # print(metric(x_new, x_core))
            if metric(x_new, x_core) < (dbscan_model.eps): # np.reshape(x_new, (1,-1)), np.reshape(x_core,(1,-1))
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

def main():
    # test()
    global  dataset, root_output, dir_tsne_plots, model_list , root_model, dbscan_eps, mac_dic

    # Parse Arguments
    parser = argparse.ArgumentParser(usage=c.PERIODIC_MOD_USAGE, add_help=False)
    parser.add_argument("-i", dest="dataset", default="")
    parser.add_argument("-o", dest="root_model", default="")
    parser.add_argument("-e", dest="eps", default=1)

    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    if args.help:
        print_usage(0)

    print("Running %s..." % c.PATH)

    # Error checking command line args

    dataset = args.dataset
    root_model = args.root_model
    dbscan_eps = args.eps
    errors = False
    #check -i in features
    if dataset != 'train' and dataset != 'test' and dataset != 'routines' and dataset != 'uncontrolled' and dataset != 'uncontrolled02':
        errors = True
        print(c.NO_FEAT_DIR, file=sys.stderr)

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

    print("Processing dataset: %s\nOutput files placed in: %s" % (dataset, root_model))
    root_output = os.path.join(root_model, 'output')
    if not os.path.exists(root_output):
        os.system('mkdir -pv %s' % root_output)
        for model_alg in model_list:
            model_dir = '%s/%s' % (root_model, model_alg)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
                
    mac_dic = utils.read_mac_address()
    train_models()


def train_models():
    global dataset, root_model, root_output, dir_tsne_plots, mac_dic
    """
    Scan feature folder for each device
    """
    print('root_model: %s' % root_model)
    print('root_output: %s' % root_output)
    lfiles = []
    lparas = []
    ldnames = []

    random_state = 422
    print("random_state:", random_state)
    for csv_file in os.listdir('data/%s-std/' % dataset):
        if csv_file.endswith('.csv'):
            print(csv_file)
            dname = csv_file[:-4]

            lparas.append((dataset, dname, random_state))
    p = Pool(num_pools)
    t0 = time.time()
    list_results = p.map(eid_wrapper, lparas)
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
    # p.map(target=eval_individual_device, args=(lfiles, ldnames))


def eid_wrapper(a):
    return eval_individual_device(a[0], a[1], a[2])


def eval_individual_device(dataset, dname, random_state, specified_models=None):
    global root_model, root_output, dbscan_eps
    """

    """
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=FutureWarning)

    """
    Prepare the directories and add only models that have not been trained yet 
    """
    model_alg = 'filter'
    model_dir = '%s/%s' % (root_model, model_alg)
    model_file = '%s/%s%s.model' % (model_dir, dname, model_alg)

    """
    Get periods from fingerprinting files
    """
    periodic_tuple = []
    host_set = set()
    with open('./period_detection/freq_period/2021_fingerprints/%s.txt' % dname, 'r') as file:
        for line in file:
            tmp = line.split()
            # print(tmp)
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
            host_set.add(tmp_host)


    """
    Load and preprocess testing data
    """

    if not os.path.isfile("data/%s-std/%s.csv" % (dataset,dname)):
        return 0

    print('loading test data %s' % dname)
    test_data = pd.read_csv("data/%s-std/%s.csv" % (dataset,dname))
    # test_data = pd.read_csv("data/test-std/%s.csv" % dname)
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
    num_of_state = len(set(y_labels_test))

    log_dir = os.path.join(root_model, '%s_logs' % dataset)
    os.system('mkdir -pv %s' % log_dir)

    host_protocol_dic = {}
    for i in range(len(test_feature)):
        if (test_hosts[i], test_protocols[i]) not in host_protocol_dic:
            host_protocol_dic[(test_hosts[i], test_protocols[i])] = 1
        else:
            host_protocol_dic[(test_hosts[i], test_protocols[i])] += 1
    with open(os.path.join(log_dir,'%s.txt' % dname),'a+') as f:
        f.write('=================\n')
        for k, v in host_protocol_dic.items():
            f.write('\n%s %s: %d\n' % (k[0], k[1], v))
        f.write('-----------------\n')
    """
    Filter local and DNS/NTP. 
    """
    filter_dns = []
    for i in range(len(test_feature)):
        if test_protocols[i] == 'DNS' or test_protocols[i] == 'MDNS' or test_protocols[i] == 'NTP' or test_protocols[i] == 'SSDP' or test_protocols[i] == 'DHCP':
            filter_dns.append(False)
        else:
            filter_dns.append(True)
    test_feature = test_feature[filter_dns]
    test_hosts = test_hosts[filter_dns]
    test_protocols = test_protocols[filter_dns]
    events = events[filter_dns]
    y_labels_test = y_labels_test[filter_dns]
    test_data_numpy = test_data_numpy[filter_dns]
    
    """
    Filter local 
    """
    local_mac_list = ['00:0c:43:26:60:00', '22:ef:03:1a:97:b9', 'ff:ff:ff:ff:ff:ff']
    filter_local = []
    for i in range(len(test_feature)):
        if test_hosts[i] in mac_dic or test_hosts[i] in local_mac_list or test_hosts[i]=='multicast' or ':' in test_hosts[i]:

            filter_local.append(False) 
        else:
            filter_local.append(True)
    test_feature = test_feature[filter_local]
    # test_timestamp = test_timestamp[filter_local]
    test_hosts = test_hosts[filter_local]
    test_protocols = test_protocols[filter_local]
    events = events[filter_local]
    y_labels_test = y_labels_test[filter_local]
    test_data_numpy = test_data_numpy[filter_local]
    """
    For each tuple: 
    """
    ret_results = []
    res_left = 0
    res_filtered = 0
    for tup in periodic_tuple:
        tmp_host = tup[0]
        tmp_proto = tup[1]
        if tmp_host == '':
            continue

        print('------%s------' %dname)
        print(tmp_proto, tmp_host)
        
        if tmp_host == 'n-devs.tplinkcloud.com':
            tmp_host_model = 'devs.tplinkcloud.com'
        else:
            tmp_host_model = tmp_host

        model_alg = 'filter'
        model_dir = os.path.join(root_model, model_alg)
        if not os.path.exists(model_dir):
            os.system('mkdir -pv %s' % model_dir)
        model_file = os.path.join(model_dir, dname + tmp_host_model + tmp_proto +".model")


        print("predicting by trained_model")
        print('Test len before:',len(test_feature))
        filter_test = []
        for i in range(len(test_feature)):
            if tmp_host.startswith('*'):
                matched_suffix = test_hosts[i].endswith(tmp_host[2:])
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

        if len(test_feature_part) == 0:
            print('test feature matched host/proto == 0')  
            continue
        print(test_feature_part.shape)

        """
        Load trained models
        """

        model = pickle.load(open(model_file, 'rb'))['trained_model']

        y_new = dbscan_predict(model,test_feature_part)

        count_left = 0
        event_after = set()
        events_tmp = set()
        state_after = set()
        filter_list = []

        if len(y_new) < 1:
            continue


        print('testing set average prediction: ',len(y_new), np.mean(y_new), np.var(y_new), np.mean(y_new) - 2 * np.var(y_new) )

        for i in range(len(y_new)):


            if y_new[i] < 0:  # <= max(np.mean(y_train) - 2 * np.var(y_train), -10 ): #  # activity
                event_after.add(events_part[i])
                state_after.add(y_labels_test_part[i])
                count_left += 1
                
                filter_list.append(True)

            else:
                filter_list.append(False)   # periodic traffic

        if len(filter_list) != len(y_new):
            print('ER')
        count_tmp = 0
        for i in range(len(filter_test)):
            if filter_test[i] == False:
                filter_test[i] = True

            elif filter_test[i] == True: # true, (proto, host)
                if filter_list[count_tmp] == False: # filter
                    filter_test[i] = False
                count_tmp += 1
            else:
                print('ER')
                exit(1)
        
        if len(filter_test) != len(test_feature):
            print('ER')


        test_feature = test_feature[filter_test]
        test_hosts = test_hosts[filter_test]
        test_protocols = test_protocols[filter_test]
        events = events[filter_test]
        y_labels_test = y_labels_test[filter_test]
        test_data_numpy = test_data_numpy[filter_test]


        
        res_left += count_left
        res_filtered += test_feature_part.shape[0] - count_left
        print("count_left" , count_left/test_feature_part.shape[0], count_left, test_feature_part.shape[0])
        # print("Activities left:", len(event_after)/len(events_tmp), len(event_after), len(events_tmp))

        print('Test len after:',len(test_feature))
        print('-------------')
        """
        Save the logs
        """
        
        with open(os.path.join(log_dir,'%s.txt' % dname),'a+') as f:
            f.write('%s %s: ' % (tmp_proto, tmp_host))
            f.write('\nFlows left: %d %d %2f\n\n' % (count_left, test_feature_part.shape[0], count_left/test_feature_part.shape[0] ))

    host_protocol_dic = {}
    for i in range(len(test_feature)):
        if ((test_hosts[i], test_protocols[i]) not in host_protocol_dic):
            host_protocol_dic[(test_hosts[i], test_protocols[i])] = 1
        else:
            host_protocol_dic[(test_hosts[i], test_protocols[i])] += 1
    with open(os.path.join(log_dir,'%s.txt' % dname),'a+') as f:
        f.write('\n')
        for k, v in host_protocol_dic.items():
            f.write('\n%s %s: %d\n' % (k[0], k[1], v))
    """
    Logging
    """
    print('Flows left: ', len(test_feature)/len_test_before,len(test_feature), len_test_before)
    print('Activity left: ',len(set(test_data_numpy[:,-4]))/num_of_event, len(set(test_data_numpy[:,-4])), num_of_event)
    with open(os.path.join(root_model,'%s_results.txt' % dataset),'a+') as f:
        f.write('%s' % dname)
        f.write('\nFlows left: %2f %d %d' % (len(test_feature)/len_test_before,len(test_feature), len_test_before))
        f.write('\nActivity left: %2f %d %d \n\n' % (len(set(test_data_numpy[:,-4]))/num_of_event, len(set(test_data_numpy[:,-4])), num_of_event))
    test_feature = pd.DataFrame(test_feature)
    test_feature['device'] = test_data_numpy[:,-6]
    test_feature['state'] = test_data_numpy[:,-5]
    test_feature['event'] = test_data_numpy[:,-4]
    test_feature['start_time'] = test_data_numpy[:,-3]
    test_feature['protocol'] = test_data_numpy[:,-2]
    test_feature['hosts'] = test_data_numpy[:,-1]
    # test_feature = pd.DataFrame(test_feature , columns=cols_feat) 

    output_dir = 'data/%s-filtered-std/' % dataset
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filtered_train_processed= '%s/%s.csv' % (output_dir , dname)
    # filtered_train_processed= 'data/test-filtered-std/%s.csv' % ( dname)
    test_feature.to_csv(filtered_train_processed, index=False)

    return 0

if __name__ == '__main__':
    main()
    num_pools = 12
