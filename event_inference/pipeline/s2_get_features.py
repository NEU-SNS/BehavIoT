import os
import sys
from multiprocessing import Process
from multiprocessing import Manager
import ipaddress
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels import robust
import Constants as c
import utils


cols_feat = [ "meanBytes", "minBytes", "maxBytes", "medAbsDev",
             "skewLength", "kurtosisLength", "meanTBP", "varTBP", "medianTBP", "kurtosisTBP",
             "skewTBP", "network_total", "network_in", "network_out", "network_external", "network_local",
            "network_in_local", "network_out_local", "meanBytes_out_external",
            "meanBytes_in_external", "meanBytes_out_local", "meanBytes_in_local", 
            "device", "state", "event", "start_time", "remote_ip", "remote_port" ,"trans_protocol", "raw_protocol", "protocol", "hosts"]
"""
INPUT: intermediate files
OUTPUT: features for models, with device and state labels 
"""

random_ratio = 0.8
num_per_exp = 10
mac_dic = {}
#is_error is either 0 or 1
def print_usage(is_error):
    print(c.GET_FEAT_USAGE, file=sys.stderr) if is_error else print(c.GET_FEAT_USAGE)
    exit(is_error)


def main():
    global mac_dic
    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % c.PATH)
    
    #error checking
    #check that there are 2 or 3 args
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(c.WRONG_NUM_ARGS % (2, (len(sys.argv) - 1)), file=sys.stderr)
        print_usage(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    str_num_proc = sys.argv[3] if len(sys.argv) == 4 else "5"

    #check in_dir
    errors = False
    if not os.path.isdir(in_dir):
        errors = True
        print(c.INVAL % ("Decoded pcap directory", in_dir, "directory"), file=sys.stderr)
    else:
        if not os.access(in_dir, os.R_OK):
            errors = True
            print(c.NO_PERM % ("decoded pcap directory", in_dir, "read"), file=sys.stderr)
        if not os.access(in_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("decoded pcap directory", in_dir, "execute"), file=sys.stderr)

    #check out_dir
    if os.path.isdir(out_dir):
        if not os.access(out_dir, os.W_OK):
            errors = True
            print(c.NO_PERM % ("output directory", out_dir, "write"), file=sys.stderr)
        if not os.access(out_dir, os.X_OK):
            errors = True
            print(c.NO_PERM % ("output directory", out_dir, "execute"), file=sys.stderr)

    #check num_proc
    bad_proc = False
    num_proc = 5
    try:
        num_proc = int(str_num_proc)
        if num_proc < 0:
            errors = bad_proc = True
    except ValueError:
        errors = bad_proc = True

    if bad_proc:
        print(c.NON_POS % ("number of processes", str_num_proc), file=sys.stderr)

    if errors:
        print_usage(1)
    #end error checking
    print("number of processes ", num_proc)
    print("Input files located in: %s\nOutput files placed in: %s\n" % (in_dir, out_dir))
    
    mac_dic = utils.read_mac_address()

    group_size = 50
    dict_dec = dict()
    dircache = os.path.join(out_dir, 'caches')
    if not os.path.exists(dircache):
        os.system('mkdir -pv %s' % dircache)
    # Parse input file names
    # in_dir/dev_dir/act_dir/dec_file
    for dev_dir in os.listdir(in_dir):
        if dev_dir.startswith("."):
            continue
        training_file = os.path.join(out_dir, dev_dir + '.csv') #Output file
        # Check if output file exists
        # if os.path.exists(training_file):
        #     print('Features for %s prepared already in %s' % (dev_dir, training_file))
        #     continue
        full_dev_dir = os.path.join(in_dir, dev_dir)
        for act_dir in os.listdir(full_dev_dir):
            full_act_dir = os.path.join(full_dev_dir, act_dir)
            if act_dir == 'unctrl' or act_dir == 'unctr'  or act_dir == 'idle':
                for dec_file in os.listdir(full_act_dir):
                    full_dec_file = os.path.join(full_act_dir, dec_file)
                    if not full_dec_file.endswith(".txt"):
                        print(c.WRONG_EXT % ("Decoded file", "text (.txt)", full_dec_file), file=sys.stderr)
                        continue
                    if not os.path.isfile(full_dec_file):
                        print(c.INVAL % ("Decoded file", full_dec_file, "file"), file=sys.stderr)
                        continue
                    if not os.access(full_dec_file, os.R_OK):
                        print(c.NO_PERM % ("decoded file", full_dec_file, "read"), file=sys.stderr)
                        continue

                    if 'companion' in dec_file:
                        state = '%s_companion_%s' % (act_dir, dev_dir)
                        device = dec_file.split('.')[-2] # the word before pcap
                    else:
                        state = act_dir
                        device = dev_dir
                        event = act_dir
                    feature_file = os.path.join(out_dir, 'caches', device + '_' + state
                                + '_' + dec_file[:-4] + '.csv') #Output cache files
                    #the file, along with some data about it
                    paras = (full_dec_file, feature_file, group_size, device, state, event)
                    #Dict contains devices that do not have an output file
                    if device not in dict_dec:
                        dict_dec[device] = []
                    dict_dec[device].append(paras)
            else:
                for event_dir in os.listdir(full_act_dir):
                    full_event_dir = os.path.join(full_act_dir, event_dir)
                    for dec_file in os.listdir(full_event_dir):
                        full_dec_file = os.path.join(full_event_dir, dec_file)
                        if not full_dec_file.endswith(".txt"):
                            print(c.WRONG_EXT % ("Decoded file", "text (.txt)", full_dec_file), file=sys.stderr)
                            continue
                        if not os.path.isfile(full_dec_file):
                            print(c.INVAL % ("Decoded file", full_dec_file, "file"), file=sys.stderr)
                            continue
                        if not os.access(full_dec_file, os.R_OK):
                            print(c.NO_PERM % ("decoded file", full_dec_file, "read"), file=sys.stderr)
                            continue

                        if 'companion' in dec_file:
                            state = '%s_companion_%s' % (act_dir, dev_dir)
                            device = dec_file.split('.')[-2] # the word before pcap
                        else:
                            state = act_dir
                            device = dev_dir
                            event = event_dir
                        feature_file = os.path.join(out_dir, 'caches', device + '_' + state
                                    + '_' + dec_file[:-4] + '.csv') #Output cache files
                        #the file, along with some data about it
                        paras = (full_dec_file, feature_file, group_size, device, state, event)
                        #Dict contains devices that do not have an output file
                        if device not in dict_dec:
                            dict_dec[device] = []
                        dict_dec[device].append(paras)

    devices = "None" if len(dict_dec) == 0 else ", ".join(dict_dec.keys())
    print("Feature files to be generated from the following devices:", devices)

    for device in dict_dec:
        training_file = os.path.join(out_dir, device + '.csv')
        list_paras = dict_dec[device]

        #create groups to run with processes
        params_arr = [ [] for _ in range(num_proc) ]

        #create results array
        results = Manager().list()

        #split pcaps into num_proc groups
        for i, paras in enumerate(list_paras):
            params_arr[i % num_proc].append(paras)

        procs = []
        for paras_list in params_arr:
            p = Process(target=run, args=(paras_list, results))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

        if len(results) > 0:
            pd_device = pd.concat(results, ignore_index=True) #Concat all cache files together
            pd_device.to_csv(training_file, index=False) #Put in CSV file
            print("Results concatenated to %s" % training_file)


def run(paras_list, results):
    for paras in paras_list:
        full_dec_file = paras[0]
        feature_file = paras[1]
        device = paras[3]
        state = paras[4]
        event = paras[5]

        tmp_data = load_features_per_exp(full_dec_file, feature_file, device, state, event)
        if tmp_data is None or len(tmp_data) == 0:
            continue
        results.append(tmp_data)


def load_features_per_exp(dec_file, feature_file, device_name, state, event):

    # Attempt to extract data from input files if not in previously-generated cache files
    feature_data = extract_features(dec_file, feature_file, device_name, state, event)
    if feature_data is None or len(feature_data) == 0: #Can't extract from input files
        print('No data or features from %s' % dec_file)
        return
    else: #Cache was generated; save to file
        pass
        # feature_data.to_csv(feature_file, index=False)
    return feature_data


#Create CSV cache file
def extract_features(dec_file, feature_file, device_name, state, event):
    col_feat = cols_feat
    pd_obj_all = pd.read_csv(dec_file, sep="\t")
    pd_obj = pd_obj_all.loc[:, :]
    num_total = len(pd_obj_all)
    if pd_obj is None or num_total < 2:
        return
    if state == "power":
        return


    # print("In decoded: %s\n  Out features: %s" % (dec_file, feature_file))
    feature_data = pd.DataFrame()


    d = compute_tbp_features(pd_obj, device_name, state, event)
    feature_data = feature_data.append(pd.DataFrame(data=[d], columns=col_feat))
    return feature_data


#Use Pandas to perform stat analysis on raw data
def compute_tbp_features(pd_obj, device_name, state, event):
    start_time = pd_obj.ts.min()
    # end_time = pd_obj.ts.max()
    # group_len = end_time - start_time
    meanBytes = pd_obj.frame_len.mean()
    minBytes = pd_obj.frame_len.min()
    maxBytes = pd_obj.frame_len.max()
    medAbsDev = robust.mad(pd_obj.frame_len)
    skewL = skew(pd_obj.frame_len)
    kurtL = kurtosis(pd_obj.frame_len)
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentiles = np.percentile(pd_obj.frame_len, p)
    kurtT = kurtosis(pd_obj.ts_delta)
    skewT = skew(pd_obj.ts_delta)
    meanTBP = pd_obj.ts_delta.mean()
    varTBP = pd_obj.ts_delta.var()
    medTBP = pd_obj.ts_delta.median()
    network_in = 0 # Network going to 192.168.10.204, or home.
    network_out = 0 # Network going from 192.168.10.204, or home.
    # network_both = 0 # Network going to/from 192.168.10.204, or home both present in source.
    
    network_external = 0 # Network not going to just 192.168.10.248.
    network_local = 0
    network_in_local = 0 # 
    network_out_local = 0 #
    # anonymous_source_destination = 0
    network_total = 0
    meanBytes_out_external = 0 
    meanBytes_in_external = 0 
    meanBytes_out_local = 0 
    meanBytes_in_local = 0 

    my_device_addr = ''
    external_destination_addr = ''
    local_destination_device = ''
    
    my_device_mac = mac_dic[device_name]
    for i, j, srcport, dstport, m in zip(pd_obj.ip_src, pd_obj.ip_dst, pd_obj.srcport, pd_obj.dstport, pd_obj.mac_addr):
        if j == '192.168.0.2':
            my_device_addr = j
            external_destination_addr = i
            remote_port = srcport
            break
        if m == my_device_mac:
            my_device_addr = j
            external_destination_addr = i
            remote_port = srcport
        else:
            my_device_addr = i
            if m in mac_dic.values(): # local 
                local_destination_device = list(mac_dic.keys())[list(mac_dic.values()).index(m)]
                external_destination_addr = ''
                remote_port = ''
            elif ipaddress.ip_address(j).is_private==True or j=="129.10.227.248" or j=="129.10.227.207":
                local_destination_device = m
                external_destination_addr = ''
                remote_port = ''
            else:
                external_destination_addr = j
                remote_port = dstport
            break


    for i, j, f_len in zip(pd_obj.ip_src, pd_obj.ip_dst, pd_obj.frame_len):
        network_total += 1
        
        if ipaddress.ip_address(i).is_private==True and (ipaddress.ip_address(j).is_private==False) and j!= "129.10.227.248" and j!= "129.10.227.207": # source addr i = 192.168.10.*, j != 192.168.10.* and != 129.10.227.248
            network_out += 1
            network_external += 1
            meanBytes_out_external += f_len
            
        elif ipaddress.ip_address(j).is_private==True and (ipaddress.ip_address(i).is_private==False) and i!= "129.10.227.248" and i!= "129.10.227.207": # destation addr j = 192.168.10.*, i != 192.168.10.* and != 129.10.227.248
            network_in += 1
            network_external += 1
            meanBytes_in_external += f_len

        # FIXME
        elif i == my_device_addr and (ipaddress.ip_address(j).is_private==True or j=="129.10.227.248" or j=="129.10.227.207"): # local out
            network_out_local += 1
            network_local += 1
            meanBytes_out_local+= f_len
        elif (ipaddress.ip_address(i).is_private==True or i=="129.10.227.248" or i=="129.10.227.207") and j == my_device_addr: #router
            network_in_local += 1
            network_local += 1
            meanBytes_in_local += f_len

        elif pd_obj.host == 'local':
            network_local += 1
            # meanBytes_in_local += f_len
        else:
            pass
            # anonymous_source_destination += 1
        
    meanBytes_out_external = meanBytes_out_external/network_out if network_out else 0
    meanBytes_in_external = meanBytes_in_external/network_in if network_in else 0
    meanBytes_out_local = meanBytes_out_local/network_out_local if network_out_local else 0
    meanBytes_in_local = meanBytes_in_local/network_in_local if network_in_local else 0

    # host is either from the host column, or the destination IP if host doesn't exist
    
    hosts = set([ str(host) for i, host in enumerate(pd_obj.host.fillna("")) ])

    
    protocol = set([str(proto) for i, proto in enumerate(pd_obj.protocol.fillna(""))])
    raw_protocol = protocol
    if ('DNS' in protocol) or ('DHCP' in protocol) or ('NTP' in protocol) or ('SSDP' in protocol) or ('MDNS' in protocol):
        pass
    else:
        if pd_obj.trans_proto[0] == 6:
            protocol = set(['TCP'])
        elif pd_obj.trans_proto[0] == 17:
            protocol = set(['UDP'])
    trans_protocol = pd_obj.trans_proto[0]
    if network_total == network_local: 
        # hosts = set(['local'])
        hosts = set([str(local_destination_device)])

    host_output = ";".join([x for x in hosts if x!= ""])
    if host_output.startswith('ec') and (host_output.endswith('compute.amazonaws.com') or host_output.endswith('compute-1.amazonaws.com')):
            host_output = '*.compute.amazonaws.com'
    if host_output == '':
        if str(external_destination_addr) == '':
            print('Error:', device_name, state, event, pd_obj.ip_src, pd_obj.ip_dst)
            exit(1)
        host_output = str(external_destination_addr)
        
    remote_ip = external_destination_addr
    
    d = [ meanBytes, minBytes, maxBytes, medAbsDev, skewL, kurtL, meanTBP, varTBP, medTBP,
         kurtT, skewT, network_total, network_in, network_out, network_external, network_local,
         network_in_local, network_out_local, meanBytes_out_external,
         meanBytes_in_external, meanBytes_out_local, meanBytes_in_local, device_name, state, event, start_time, 
         remote_ip, remote_port ,trans_protocol, raw_protocol, ";".join([x for x in protocol if x!= ""]), host_output ]

    return d


if __name__ == '__main__':
    main()

