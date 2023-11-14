import warnings
import os
import sys
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import time
from multiprocessing import Pool
import Constants as c

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

num_pools = 1

root_output = ''
root_feature = ''
root_model = ''


def main():
    global  root_output, model_list , root_feature, root_model

    # Parse Arguments
    parser = argparse.ArgumentParser(usage=c.PREDICT_MOD_USAGE, add_help=False)
    parser.add_argument("-i", dest="root_feature", default="")
    parser.add_argument("-o", dest="root_model", default="")
    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    # if args.help:
    #     print_usage(0)

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

    if errors:
        print('Errorrrr')
        exit(1)

    # end error checking

    print("Input files located in: %s \n Output files placed in: %s" % (root_feature, root_model))
    # root_output = os.path.join(root_model, 'fingerprint')
    root_output = root_model
    if not os.path.exists(root_output):
        os.system('mkdir -pv %s' % root_output)


    train_models()


def train_models():
    # global root_feature, root_model, root_output
    """
    Scan feature folder for each device
    """
    print('root_feature: %s' % root_feature)
    print('root_output: %s' % root_output)
    lfiles = []
    lparas = []
    ldnames = []

    # set a random state
    # random_state = random.randint(0, 1000)
    random_state = 422
    print("random_state:", random_state)
    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            print(csv_file)
            train_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]
            lparas.append((train_data_file, dname, random_state))
    p = Pool(num_pools)
    t0 = time.time()
    list_results = p.map(eid_wrapper, lparas) 
            
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
    return fingerprint_individual_device(a[0], a[1], a[2])


def fingerprint_individual_device(train_data_file, dname, random_state):
    """
    fingerprinting
    INPUT: 
        train_data_file, dname, random_state
    RETURN:
        result_file, result_dict
    """

    """
    Read training file
    """
    train_data = pd.read_csv(train_data_file)
    num_data_points = len(train_data)
    if num_data_points < 1:
        print('  Not enough data points for %s, skipping' % dname)
        return
    print('\t# Total data points for %s: %d ' % (dname, num_data_points))

    train_hosts = np.array(train_data['hosts'].fillna('').values)
    train_protocol = np.array(train_data['protocol'].fillna('').values)
    train_labels = np.array(train_data.state)
    train_event = np.array(train_data.event)
    train_start_time = np.array(train_data.start_time)
    
    for i in range(len(train_protocol)):
        if 'TCP' in train_protocol[i]:
            train_protocol[i] = 'TCP'
        elif 'UDP' in train_protocol[i]:
            train_protocol[i] = 'UDP'
        elif 'TLS' in train_protocol[i]:
            train_protocol[i] = 'TLS'
        if ';' in train_protocol[i]:
            tmp = train_protocol[i].split(';')
            train_protocol[i] = ' & '.join(tmp)
    # print(train_hosts)
    domain_set = set()
    for i in range(len(train_hosts)):
        if train_hosts[i] != '' and train_hosts[i] != None:
            try:
                tmp = train_hosts[i].split(';')
            except:
                print(train_hosts[i])
                exit(1)
            train_hosts[i] = tmp[0]
        if train_hosts[i] == None:
            train_hosts[i] == 'non'
        train_hosts[i] = train_hosts[i].lower()

        domain_set.add(train_hosts[i])


    for i in domain_set.copy():
        matched = 0
        if len(i.split('.')) >= 4: # a.b.c.d -> *.b.c.d
            suffix = '.'.join(i.split('.')[-3:]) # b.c.d
            for j in domain_set.copy(): 
                if j == i or j.startswith('*'):
                    continue
                elif j.endswith(suffix):
                    matched = 1
                    domain_set.remove(j)

            if matched == 1:
                domain_set.remove(i)
                # print('Remove : ',i)
                domain_set.add('*.'+suffix)

    # return 0

    lb = LabelBinarizer()
    lb.fit(train_labels)  
    # set of all labels
    positive_label_set = lb.classes_.tolist()

    res_dict = {}
    
    for l in positive_label_set: 
        res_dict[l] = {}
        cur_label_num = set()
        for i in range(len(train_event)):
            if train_labels[i] == l:
                cur_label_num.add(train_event[i])

        for i in range(len(train_labels)):
            if train_labels[i] == l:
                if len(train_hosts[i].split('.')) >=4 and ('*.'+'.'.join(train_hosts[i].split('.')[-3:]) in domain_set):
                    tuple_pair = ('*.'+'.'.join(train_hosts[i].split('.')[-3:]), train_protocol[i])
                    train_hosts[i] = '*.'+'.'.join(train_hosts[i].split('.')[-3:])
                else:
                    tuple_pair = (train_hosts[i] if train_hosts[i]!="" else 'none', train_protocol[i])
                if tuple_pair in res_dict[l]:
                    res_dict[l][tuple_pair][0] += 1
                else:
                    res_dict[l][tuple_pair] = [1,len(cur_label_num)]

    output_file = "%s/%s.txt" %(root_output, dname)

    if res_dict == 0 or res_dict is None or len(res_dict) == 0: return 0

     # res_dict: Key: activity, Value: dictionary{key: tuple_pair, value: [num of flows, num of activity events]}
    tmp_outfile = output_file
    tmp_res = res_dict
    domain_list = set()
    domain_dict = {} # k: activity name, v: (host, protocol)
    if os.path.isfile(tmp_outfile):
        os.remove(tmp_outfile)
    with open(tmp_outfile, 'w+') as off:
        # off.write('random_state:',random_state)
        for k,v in tmp_res.items():
            off.write('Activity: %s: \n' % k)
            activity_domain_list = set()
            for tuple_pair in v.keys():
                off.write('%s, %s, %d, %d \n' %(tuple_pair[0], tuple_pair[1], int(v[tuple_pair][0]), int(v[tuple_pair][1])))
                if int(v[tuple_pair][0]) / int(v[tuple_pair][1]) >= 0.9:
                    domain_list.add(tuple_pair)
                    activity_domain_list.add(tuple_pair)
            if len(activity_domain_list)==0:
                for tuple_pair in v.keys():
                    if int(v[tuple_pair][0]) / int(v[tuple_pair][1]) >= 0.8:
                        domain_list.add(tuple_pair)
                        activity_domain_list.add(tuple_pair)
            off.write('\n')
            domain_dict[k] = []
            for i in activity_domain_list:
                domain_dict[k].append(str(i[0])+','+str(i[1]))
            off.write('\n')
        off.write('\nDomain List:\n' )
        for i in domain_list:
            off.write('%s, %s\n'% (i[0], i[1]))
        off.write('\n')
        for k,v in domain_dict.items():
            
            event_ts_dic = {}
            for i in range(len(train_labels)):
                if train_labels[i] == k:
                    if ((str(train_hosts[i])+','+str(train_protocol[i])) in v):
                        if train_event[i] not in event_ts_dic:
                            event_ts_dic[train_event[i]] = [float(train_start_time[i])]
                        else:
                            event_ts_dic[train_event[i]].append(float(train_start_time[i]))
            
            t_delta_ave = []
            for e in event_ts_dic.keys():
                t_delta = np.max(event_ts_dic[e]) - np.min(event_ts_dic[e])
                t_delta_ave.append(t_delta)
            average = np.mean(t_delta_ave) if len(t_delta_ave) > 0 else -1
            

            off.write('fingerprint- %s: %s\n'% (k, ';'.join(v)))

            try:
                off.write('ts- %s:%.2f\n'% (k, average))
            except:
                print(print(k, t_delta_ave))
                exit(1)


    
    return output_file, res_dict
   

if __name__ == '__main__':
    main()
    num_pools = 1
