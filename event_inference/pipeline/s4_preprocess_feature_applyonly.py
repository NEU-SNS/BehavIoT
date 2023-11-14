import warnings
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import Constants as c
import utils

warnings.simplefilter("ignore", category=DeprecationWarning)


num_pools = 12
cols_feat = [ "meanBytes", "minBytes", "maxBytes", "medAbsDev",
             "skewLength", "kurtosisLength", "meanTBP", "varTBP", "medianTBP", "kurtosisTBP",
             "skewTBP", "network_total", "network_in", "network_out", "network_external", "network_local",
            "network_in_local", "network_out_local", "meanBytes_out_external",
            "meanBytes_in_external", "meanBytes_out_local", "meanBytes_in_local", "device", "state", "event", "start_time", "protocol", "hosts"]
            


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
    # test()
    global  root_output , root_feature, root_model, root_test, root_test_out

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
    global root_feature, root_model, root_output, dir_tsne_plots
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
    # p.map(target=eval_individual_device, args=(lfiles, ldnames))


def eid_wrapper(a):
    return eval_individual_device(a[0], a[1], a[2])


def eval_individual_device(data_file, dname):
    global root_feature, root_model, root_test, root_test_out


    train_std_dir = '%s%s' % (root_model[:-1], '-std' )
    std_train_file = '%s/%s.csv' % (train_std_dir, dname)
    # train_pca_dir = '%s%s' % (root_model[:-1], '-pca' )
    # pca_train_file = '%s/%s.csv' % (train_pca_dir, dname)


        
    if not os.path.exists(train_std_dir):
        os.system('mkdir -pv %s' % train_std_dir)
    # if not os.path.exists(train_pca_dir):
    #     os.mkdir(train_pca_dir)
   
    train_data = pd.read_csv(data_file)

    # unctrl_data = pd.read_csv(unctrl_file)
    train_data = train_data.loc[(train_data['start_time'] >= 1638680400)] # 
    num_data_points = len(train_data)
    if num_data_points < 1:
        print('  Not enough data points for %s' % dname)
        return
    print('\t#Total data points: %d ' % num_data_points)
    

    X_feature = train_data.drop(['device', 'state', 'event' ,'start_time', 'protocol', 'hosts'], axis=1).fillna(-1)
    train_length = len(X_feature)


    X_feature = np.array(X_feature)


    '''
    Load ss and pca
    '''
    model_path = './model/SS_PCA'
    # saved_dictionary = dict({'ss':ss,'pca':pca})
    model_file = "%s/%s.pkl"%(model_path,dname)


    try:
        models = pickle.load(open(model_file, 'rb'))
        ss = models['ss']

        test_data_std = ss.transform(X_feature)

    except:
        ss= StandardScaler()
        # pca = PCA(n_components=10)
        test_data_std = ss.fit_transform(X_feature)
        # test_data_pca = pca.fit_transform(test_data_std)
        saved_dictionary = dict({'ss':ss})
        pickle.dump(saved_dictionary, open("%s/%s.pkl"%(model_path,dname),"wb"))


    test_data_std = pd.DataFrame(test_data_std, columns=cols_feat[:-6])
    test_data_std['device'] = np.array(train_data.device)
    test_data_std['state'] = np.array(train_data.state)
    test_data_std['event'] = np.array(train_data.event)
    test_data_std['start_time'] = np.array(train_data.start_time)
    test_data_std['protocol'] = np.array(train_data.protocol)
    test_data_std['hosts'] = np.array(train_data.hosts)


    test_data_std.to_csv(std_train_file, index=False)



if __name__ == '__main__':
    main()
    num_pools = 12

# 