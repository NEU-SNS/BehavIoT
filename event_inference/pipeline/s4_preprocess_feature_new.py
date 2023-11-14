import warnings
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import utils
from sklearn.preprocessing import StandardScaler
import time
from multiprocessing import Pool
import Constants as c
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

num_pools = 12
cols_feat = utils.get_features()
            


model_list = []
root_output = ''
dir_tsne_plots = ''
root_feature = ''
root_model = ''
root_test= ''
root_test_out = ''
#is_error is either 0 or 1
def print_usage(is_error):
    print(c.PREPRO_USAGE, file=sys.stderr) if is_error else print(c.PREPRO_USAGE)
    exit(is_error)


def main():

    global  root_output, dir_tsne_plots , root_feature, root_model, root_test, root_test_out

    # Parse Arguments
    parser = argparse.ArgumentParser(usage=c.PREPRO_USAGE, add_help=False)
    parser.add_argument("-i", dest="root_feature", default="")
    parser.add_argument("-o", dest="root_pre_train", default="")
    parser.add_argument("-h", dest="help", action="store_true", default=False)
    args = parser.parse_args()

    if args.help:
        print_usage(0)

    print("Running %s..." % c.PATH)

    # Error checking command line args
    root_feature = args.root_feature
    root_model = args.root_pre_train

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
    root_output = os.path.join(root_model, 'output')


    train_models()


def train_models():
    global root_feature, root_model, root_output
    """
    Scan feature folder for each device
    """
    print('root_feature: %s' % root_feature)
    print('root_model: %s' % root_model)
    print('root_output: %s' % root_output)

    lparas = []

    random_state = 422
    print("random_state:", random_state)
    for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            print(csv_file)
            idle_data_file = '%s/%s' % (root_feature, csv_file)
            dname = csv_file[:-4]

            lparas.append((idle_data_file, dname, random_state))
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
    return eval_individual_device(a[0], a[1], a[2])


def eval_individual_device(idle_data_file, dname):
    global root_feature, root_model, root_test, root_test_out

    
    # dirctories 
    train_feature_dir = '%s%s' % (root_feature[0:5], 'train-features') 
    test_feature_dir = '%s%s' % (root_feature[0:5], 'test-features') 
    test_std_dir = '%s%s%s' % (root_model[0:5], 'test' , '-std')
    test_pca_dir = '%s%s%s' % (root_model[0:5], 'test' , '-pca')
    # print('Test feature:',test_feature_dir)

    # train data file, std&pca files
    train_data_file = '%s/%s.csv' %(train_feature_dir, dname)
    train_std_dir = '%s%s' % (root_model[:-1], '-std' )
    std_train_file = '%s/%s.csv' % (train_std_dir, dname)
    train_pca_dir = '%s%s' % (root_model[:-1], '-pca' )
    pca_train_file = '%s/%s.csv' % (train_pca_dir, dname)

    # test data file, std&pca files
    test_file = '%s/%s.csv' % (test_feature_dir, dname)
    std_test_file = '%s/%s.csv' % (test_std_dir, dname)
    pca_test_file = '%s/%s.csv' % (test_pca_dir, dname)

    # idle_file = './data/idle-2021-features/%s.csv' % dname

    routines_file = './data/trace-features/%s.csv' % dname
    if not os.path.isfile(routines_file):
        with_routines = False
    else:
        with_routines = True
        routines_data = pd.read_csv(routines_file)

        
    if not os.path.exists(train_std_dir):
        os.mkdir(train_std_dir)

    if not os.path.exists(test_std_dir):
        os.mkdir(test_std_dir)

    
    # idle dirctories
    train_idle_std_dir = './data/idle-2021-train-std'
    train_idle_pca_dir = './data/idle-2021-train-pca'
    test_idle_std_dir = './data/idle-2021-test-std'
    test_idle_pca_dir = './data/idle-2021-test-pca'
    if not os.path.exists(train_idle_std_dir):
        os.mkdir(train_idle_std_dir)
    # if not os.path.exists(train_idle_pca_dir):
    #     os.mkdir(train_idle_pca_dir)
    if not os.path.exists(test_idle_std_dir):
        os.mkdir(test_idle_std_dir)
    # if not os.path.exists(test_idle_pca_dir):
    #     os.mkdir(test_idle_pca_dir)

    # idle std&pca files
    train_idle_std_file = '%s/%s.csv' % ( train_idle_std_dir, dname) 
    train_idle_pca_file = '%s/%s.csv' % ( train_idle_pca_dir, dname) 
    test_idle_std_file = '%s/%s.csv' % ( test_idle_std_dir, dname) 
    test_idle_pca_file = '%s/%s.csv' % ( test_idle_pca_dir, dname) 

    if with_routines:
        routines_std_dir = './data/routines-std'
        routines_pca_dir = './data/routines-pca'
        routines_std_file = '%s/%s.csv' % ( routines_std_dir, dname) 
        routines_pca_file = '%s/%s.csv' % ( routines_pca_dir, dname) 
        if not os.path.exists(routines_std_dir):
            os.mkdir(routines_std_dir)
        # if not os.path.exists(routines_pca_dir):
        #     os.mkdir(routines_pca_dir)


    if not os.path.isfile(idle_data_file):
        print('%s idle do not exist' % dname)
        return

    only_idle = False
    if not os.path.isfile(train_data_file):
        print(train_data_file)
        # no labeled data file. 
        only_idle = True 
        print('Only Idle: ', dname)
    else:
        train_data = pd.read_csv(train_data_file)
        if dname == 'ikettle':
            test_data = pd.read_csv(train_data_file)
        else:
            test_data = pd.read_csv(test_file)

    
        X_feature = train_data.drop(['device', 'state', 'event' ,'start_time', "remote_ip", "remote_port" ,"trans_protocol", "raw_protocol", 'protocol', 'hosts'], axis=1).fillna(-1)
        train_length = len(X_feature)
        test_data_feature = test_data.drop(['device', 'state', 'event','start_time', "remote_ip", "remote_port" ,"trans_protocol", "raw_protocol", 'protocol', 'hosts'], axis=1).fillna(-1)

        
    # read idle files
    idle_data = pd.read_csv(idle_data_file)
    if dname=='govee-led1' or dname=='philips-bulb':
            pass
    else:
        idle_data = idle_data.loc[(idle_data['start_time'] > 1630688400)]
        idle_data = idle_data.loc[(idle_data['start_time'] < 1631120400)]
    

    # Idle train test split: 
    if np.min(idle_data['start_time']) <= 1630698400 and np.max(idle_data['start_time']) >= 1631110000:
        split_time = 1631034000
    else: 
        split_time =  np.max(idle_data['start_time']) - (np.max(idle_data['start_time']) - np.min(idle_data['start_time']))/5 
    train_idle_data = idle_data.loc[(idle_data['start_time'] < split_time)]  #  1556420400
    test_idle_data = idle_data.loc[(idle_data['start_time'] >= split_time)] 


    train_idle_feature = train_idle_data.drop(['device', 'state', 'event','start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    test_idle_feature = test_idle_data.drop(['device', 'state', 'event','start_time', 'protocol', 'hosts'], axis=1).fillna(-1)

    # unctrl_data = pd.read_csv(unctrl_file)
    
    
    # test_length = test_data.shape[0]
    
    num_data_points = len(idle_data)
    if num_data_points < 1:
        print('  Not enough data points for %s' % dname)
        return
    # print('\t#Total data points: %d ' % num_data_points)
    
    
    if with_routines:
        routines_feature = routines_data.drop(['device', 'state', 'event','start_time', 'protocol', 'hosts'], axis=1).fillna(-1)

    
    print('train test idle:', dname, len(train_idle_data), len(test_idle_data))
    if len(train_idle_data)==0 or len(test_idle_data)==0:
        print('Not enough idle data points for:',dname,len(train_idle_data), len(test_idle_data))
        return
    train_idle_len = len(train_idle_feature)

    if only_idle:
        X_feature =  np.array(train_idle_feature)
    else:
        X_feature = pd.concat([X_feature, train_idle_feature])
        X_feature = np.array(X_feature)

    if len(X_feature) <=0:
        print(len(X_feature),dname)
        print('No data')
        exit(1)
    

    ss= StandardScaler()

    # ss
    X_all_std = ss.fit_transform(X_feature)
    if not only_idle:
        test_data_std = ss.transform(test_data_feature)
    test_idle_std = ss.transform(test_idle_feature)
    



    if with_routines:
        routines_std = ss.transform(routines_feature)
        # routines_pca = pca.transform(routines_std)

    '''
    Save ss and pca
    '''
    model_path = './model/SS_PCA'
    saved_dictionary = dict({'ss':ss}) # ,'pca':pca
    pickle.dump(saved_dictionary, open("%s/%s.pkl"%(model_path,dname),"wb"))

    
    if only_idle:
        X_idle_std = X_all_std
        # X_idle_pca = X_all_pca
    else:
        X_std = X_all_std[:train_length,:] 
        X_idle_std = X_all_std[train_length:,:] 
        # X_pca = X_all_pca[:train_length,:] 
        # X_idle_pca = X_all_pca[train_length:,:]  # .iloc

        X_feature_std = pd.DataFrame(X_std, columns=cols_feat[:-6])
        X_feature_std['device'] = train_data.device
        X_feature_std['state'] = train_data.state
        X_feature_std['event'] = train_data.event
        X_feature_std['start_time'] = train_data.start_time
        X_feature_std['protocol'] = train_data.protocol
        X_feature_std['hosts'] = train_data.hosts


        test_data_std = pd.DataFrame(test_data_std, columns=cols_feat[:-6])
        test_data_std['device'] = test_data.device
        test_data_std['state'] = test_data.state
        test_data_std['event'] = test_data.event
        test_data_std['start_time'] = test_data.start_time
        test_data_std['protocol'] = test_data.protocol
        test_data_std['hosts'] = test_data.hosts



        X_feature_std.to_csv(std_train_file, index=False)
        test_data_std.to_csv(std_test_file, index=False)



    X_idle_std = pd.DataFrame(X_idle_std, columns=cols_feat[:-6])
    X_idle_std['device'] = np.array(train_idle_data.device)
    X_idle_std['state'] = np.array(train_idle_data.state)
    X_idle_std['event'] = np.array(train_idle_data.event)
    X_idle_std['start_time'] = np.array(train_idle_data.start_time)
    X_idle_std['protocol'] = np.array(train_idle_data.protocol)
    X_idle_std['hosts'] = np.array(train_idle_data.hosts)

    

    test_idle_std = pd.DataFrame(test_idle_std, columns=cols_feat[:-6])
    test_idle_std['device'] = np.array(test_idle_data.device)
    test_idle_std['state'] = np.array(test_idle_data.state)
    test_idle_std['event'] = np.array(test_idle_data.event)
    test_idle_std['start_time'] = np.array(test_idle_data.start_time)
    test_idle_std['protocol'] = np.array(test_idle_data.protocol)
    test_idle_std['hosts'] = np.array(test_idle_data.hosts)
    # if dname =='google-home-mini':
    #     print('Length check, train idle: ', len(X_idle_std), len(train_idle_data))
    #     print('Length check, test idle: ', len(test_idle_std), len(test_idle_data))
    if with_routines:
        routines_std = pd.DataFrame(routines_std, columns=cols_feat[:-6])
        routines_std['device'] = routines_data.device
        routines_std['state'] = routines_data.state
        routines_std['event'] = routines_data.event
        routines_std['start_time'] = routines_data.start_time
        routines_std['protocol'] = routines_data.protocol
        routines_std['hosts'] = routines_data.hosts

        routines_std.to_csv(routines_std_file, index=False)


    
    X_idle_std.to_csv(train_idle_std_file, index=False)
    test_idle_std.to_csv(test_idle_std_file, index=False)
    # unctrl_std.to_csv(unctrl_std_file, index=False)

    """

    X_idle_pca = pd.DataFrame(X_idle_pca)
    X_idle_pca['device'] = np.array(train_idle_data.device)
    X_idle_pca['state'] = np.array(train_idle_data.state)
    X_idle_pca['event'] = np.array(train_idle_data.event)
    X_idle_pca['start_time'] = np.array(train_idle_data.start_time)
    X_idle_pca['protocol'] = np.array(train_idle_data.protocol)
    X_idle_pca['hosts'] = np.array(train_idle_data.hosts)

    

    test_idle_pca = pd.DataFrame(test_idle_pca)
    test_idle_pca['device'] = np.array(test_idle_data.device)
    test_idle_pca['state'] = np.array(test_idle_data.state)
    test_idle_pca['event'] = np.array(test_idle_data.event)
    test_idle_pca['start_time'] = np.array(test_idle_data.start_time)
    test_idle_pca['protocol'] = np.array(test_idle_data.protocol)
    test_idle_pca['hosts'] = np.array(test_idle_data.hosts)

    if with_routines:
        routines_pca = pd.DataFrame(routines_pca)
        routines_pca['device'] = routines_data.device
        routines_pca['state'] = routines_data.state
        routines_pca['event'] = routines_data.event
        routines_pca['start_time'] = routines_data.start_time
        routines_pca['protocol'] = routines_data.protocol
        routines_pca['hosts'] = routines_data.hosts

        routines_pca.to_csv(routines_pca_file, index=False)
    # unctrl_pca = pd.DataFrame(unctrl_pca)
    # unctrl_pca['device'] = unctrl_data.device
    # unctrl_pca['state'] = unctrl_data.state
    # unctrl_pca['event'] = unctrl_data.event
    # unctrl_pca['start_time'] = unctrl_data.start_time
    # unctrl_pca['protocol'] = unctrl_data.protocol
    # unctrl_pca['hosts'] = unctrl_data.hosts

    
    X_idle_pca.to_csv(train_idle_pca_file, index=False)
    test_idle_pca.to_csv(test_idle_pca_file, index=False)

    # unctrl_pca.to_csv(unctrl_pca_file, index=False)
    """



if __name__ == '__main__':
    main()
    num_pools = 12

# 