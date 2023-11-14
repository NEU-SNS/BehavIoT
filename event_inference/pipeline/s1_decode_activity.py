import sys
import os
import utils
from utils import is_local, validate_ip_address
from multiprocessing import Process
import ipaddress
import numpy as np
from subprocess import Popen, PIPE
import Constants as c
import pickle


def print_usage(is_error):
    print(c.DEC_RAW_USAGE, file=sys.stderr) if is_error else print(c.DEC_RAW_USAGE)
    exit(is_error)

mac_dic = {}


def extract_host_new(ip_src, ip_dst, ip_host, count_dic, cur_time, whois_list):
    # ip_spl = ip_dst.split(".")
    host = 0
    # If having record in ip_host dict

    # multicast
    if ipaddress.ip_address(ip_dst).is_multicast: 
        host = 'Multicast'

    # local traffic
    elif is_local(ip_src, ip_dst):
        host = "local"
        count_dic['local'] += 1
    
    # 
    elif ipaddress.ip_address(ip_dst).is_private:
        host = ""

    # outbound traffic and ip_dst is in ip_host dictionary
    elif ip_dst in ip_host: 

        # if ip_host is manually added or empty
        if ip_host[ip_dst] == "" or isinstance(ip_host[ip_dst], list) == False: # 

            host = ip_host[ip_dst]

            if host == "": # 
                count_dic['blank'] += 1
            elif ip_dst in whois_list:
                count_dic['whois'] += 1
                # print('----------WHOIS:',host)
            else:
                count_dic['prior'] += 1

        # having dns or tls record 
        else:   
            for domain_time_tuple in sorted(ip_host[ip_dst],key= lambda t:t[1]):
                if cur_time >= domain_time_tuple[-1]:
                    host = domain_time_tuple[0] # ip_host[ip_dst] 
                else:
                    break

            if host == 0:
                count_dic['after'] += 1
                host = sorted(ip_host[ip_dst],key= lambda t:t[1])[0][0]
            else:
                count_dic['prior'] += 1

    # outbound traffic and ip_dst is NOT in ip_host dictionary
    else:
        # use dig -x to extract hostname
        try:
            dig = utils.dig_x(ip_dst)
            if dig is None or dig == '':
                host = ip_host[ip_dst] = ""
                # print('----------WHOIS:', ip_dst)
                count_dic['blank'] += 1
            else:
                host = ip_host[ip_dst] = dig.lower()[:-1] #  = ip_host[ip_dst] 
                # print('----------WHOIS:', host)
                whois_list.add(ip_dst)
                count_dic['whois'] += 1

        except Exception as e:
            print('Exception: ',str(e))
            host = ip_host[ip_dst] = ""
            count_dic['blank'] += 1
    return host


def extract_pcap(in_pcap, out_txt, dev_name, ip_host):

    # missing DNS records
    ip_host["8.8.8.8"] = "dns.google" # whois can't resolve this
    ip_host["155.33.33.75"] = "neu.edu"
    ip_host["155.33.33.70"] = "neu.edu"



    command = ["tshark", "-r", in_pcap, 
                # "-Y", "not tcp.analysis.duplicate_ack and not tcp.analysis.retransmission and not tcp.analysis.fast_retransmission",
                "-Tfields",
                "-e", "frame.number",
                "-e", "frame.time_epoch",
                "-e", "frame.time_delta",
                "-e", "frame.len",
                "-e", "_ws.col.Protocol",
                "-e", "tcp.stream",
                "-e", "udp.stream",
                "-e", "ip.src",
                "-e", "ip.dst",
                "-e", "tcp.srcport",
                "-e", "udp.srcport",
                "-e", "tcp.dstport",
                "-e", "udp.dstport",
                "-e", "ip.proto",
                "-e", "eth.dst",
                "-e", "_ws.expert"]
    result = []
    # Call Tshark on packets
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    # Get output. Give warning message if any
    out, err = process.communicate()
    if err:
        print("Error reading file: '{}'".format(err.decode('utf-8')))

    count_dic = {'prior':0,'after':0,'whois':0,'blank':0,'local':0}
    whois_list = set()
    my_device_mac =  mac_dic[dev_name]
    # Parsing packets
    for packet in filter(None, out.decode('utf-8').split('\n')):
        packet = np.array(packet.split())
        if packet[4] == 'ADwin' and packet[5] == 'Config':
            packet = np.delete(packet, 5)
        # add host to rest of output: 1) get host from tshark 2) get host from whois 3) host is ""
        if len(packet) > 12:
            packet = np.append(packet[:12], ' '.join(packet[12:]))
        else:
            packet = np.append(packet, '')
        # filter out layer 2 layer 3 packets
        if len(packet) < 13 or len(packet[6])>=18 or packet[4] == 'ICMP':
            # print(len(packet),packet)
            continue

        ip_src = packet[6]
        ip_dst = packet[7] # desintation host -> -e ip.dst

        if validate_ip_address(ip_src)==False or validate_ip_address(ip_dst)==False:
            continue
        cur_time = packet[1]
        # ip_spl = ip_dst.split(".")

        if my_device_mac == packet[11]:  # inbound traffic
            host = extract_host_new(ip_dst, ip_src, ip_host, count_dic, cur_time, whois_list)
        else:   # extract hostname for all outbound traffic
            host = extract_host_new(ip_src, ip_dst, ip_host, count_dic, cur_time, whois_list)


        host = host.lower()
        packet = np.append(packet,host) #append host as last column of output
        if len(packet) < 14:
            print('Length incorrect! ', packet)
            continue
        # print(len(packet), packet)
        result.append(packet)
    result = np.asarray(result)
    if len(result) == 0:
        print('len(result) == 0')
        return count_dic
    # print(dev_name, count_dic)

    # #write output file
    
    
    # ploting()
    # exit(1)

    # Partition into burst
    dev_burst = {}
    burst_threshold = 1
    if dev_name in dev_burst:
        burst_threshold = dev_burst[dev_name]

    # print('Results:', result.shape)

    flow_dic = extract_single(result)

    burst_dic = split(flow_dic, burst_threshold)
    if burst_dic == -1:
        return count_dic

    header = "frame_num\tts\tts_delta\tframe_len\tprotocol\tstreamID\tip_src\tip_dst\tsrcport\tdstport\ttrans_proto\tmac_addr\thost\n"
    count = 0
    # print("In pcap: %s\n  Out decoded: %s" % (in_pcap, out_txt))
    
    flow_count = 0
    for k, v in burst_dic.items():
        if len(k) < 5:
            # print('continued:',in_pcap)
            continue
        count += 1
        flow_count += len(v)
        for ts, flow_burst in v.items():
            if len(flow_burst) < 2:
                continue
            flow_dir = out_txt[:-4]+"/"
            
            flow_file = out_txt[:-4]+"/flow{}_burst{}.txt".format(k[3],ts)
            if not os.path.isdir(flow_dir):
                os.system("mkdir -pv %s" % flow_dir)
            with open(flow_file, "w") as f:
                f.write(header)
                for row in flow_burst:
                    try: 
                        f.write(('\t'.join(['%s']*row.size)+'\n') % tuple(row))
                    except:
                        print(row)
                        return 1
    # print("Burst num:", len(burst_list))

    # print('Packet, flow_dic, burst dic, flow_burst:',len(result), len(flow_dic), count, flow_count)

    return count_dic


def extract_single(pcap):
        """Extract flows from single burst.
            Parameters
            ----------
            pcap : np.array of shape=(n_samples, n_features)
                Numpy array containing packets of burst.
            Returns
            -------
            result : dict
                Dictionary of 5_tuple -> packets
                5 tuple is defined as (trans_proto, src, sport, dst, dport)

            """
        # Initialise result
        result = dict()

        # Extract burst timestamp
        # timestamp = burst[0, 1]

        # Loop over packets in burst
        for packet in pcap:
            # Define key as 5-tuple (trans_proto, src, sport, dst, dport)
            key = key_generate(packet)
            if key[0] == 0:
                # print(packet)
                continue

            # Add length of packet to flow
            result[key] = result.get(key, [])
            result[key].append(packet)

        # Convert lengths to numpy array.
        result = {k: np.array(v) for k, v in result.items()}

        # remove retrans and dup
        result = rm_retrans_dup(result)

        # Return result
        return result

def rm_retrans_dup(results):
    """Remove retransmission and duplicated packets and their impact to timing
    Parameters
    ----------
    results : Dictionary of 5_tuple -> packets
        5 tuple is defined as (trans_proto, src, sport, dst, dport)
    Returns
    -------
    results : 
        Dictionary of 5_tuple -> packets
        5 tuple is defined as (trans_proto, src, sport, dst, dport)

    """
    for k in results.keys():
        ts = results[k][:, 1]
        ts = ts.astype(np.float)
        # diff = results[k][:, 2]
        diff = [0] + [(ts[i + 1] - ts[i]) for i in range(len(ts)-1)]
        diff = np.array(diff)

        # print('Diff: ', diff)
        results[k][:, 2] = np.round(diff, 6)

        filter_retrans = []
        for l in range(len(results[k])):
            packet = results[k][l]

            if len(packet) != 14:
                filter_retrans.append(True)
                continue
            expert_info = packet[-2]
            if expert_info != "" and ('retransmission' in expert_info or 'Duplicate ACK' in expert_info):
                # print(packet)
                # flagged = 1
                # ts = packet[1]
                filter_retrans.append(False)
            else:
                filter_retrans.append(True)
            
        
        results[k] = results[k][filter_retrans]
        results[k] = np.delete(results[k],-2,1)
    
    return results

def key_generate(packet):
        """Extract the key of a packet and check whether it is incoming or
            outgoing.
            Parameters
            ----------
            # timestamp : float
            #     Timestamp of burst.
            packet : np.array of shape=(n_features)
        
            Returns
            -------
            key : tuple
                Key 5-tuple of flow.
            # incoming : boolean
            #     Boolean indicating whether flow is incoming.
            """
        # Define key as 5-tuple (trans_proto, src, sport, dst, dport)
        try:
            key = (packet[10], ipaddress.ip_address(packet[6]), packet[8],
                        ipaddress.ip_address(packet[7]), packet[9]) #  
        except:
            return (0,0,0,0,0) # 

        # Check if flow message is incoming
        if key[3].is_private and (key[1].is_private == False):
            incoming = True
            key = (key[0], key[3], key[4], key[1], key[2])
        elif key[3].is_private and key[1].is_private:
            if key[3] > key[1]:
                key = (key[0], key[3], key[4], key[1], key[2])
        
        # Set IP addresses to string
        key = (key[0], str(key[1]), key[2], str(key[3]), key[4])

        # Return result
        return key

def split(flow_dic, threshold=1):
        """Split packets in bursts based on given threshold.
            A burst is defined as a period of inactivity specified by treshold.
            Parameters
            ----------
            flow_dic : dict, key: 5tuple, value: packets
            threshold : float, default=1
                Burst threshold in seconds.
            Returns
            -------
            result : dict{ key: 5tuple, value: dict{key: ts, value: packets} }
                List of np.array, where each list entry are the packets in a
                burst.
            """
        # Initialise result
        result = {}
        for k, v in flow_dic.items():
            
            # Compute difference between packets
            if len(v) < 2:
                continue
            # try:
            ts = v[:, 1]
            diff = v[:, 2]
            # ts = ts.astype(np.float)
            try:
                diff = diff.astype(np.float)
                diff = np.array(diff)
            except ValueError:
                print('Time diff error: ', k, ts, diff, v[:,-1])
                exit(1)


            result[k] = result.get(k,{})
            
            # Select indices where difference is greater than threshold
            indices_split = np.argwhere(diff > threshold)
            # Add 0 as start and length as end index
            indices_split = [0] + list(indices_split.flatten()) + [v.shape[0]]
            for start, end in zip(indices_split, indices_split[1:]):
                if end == 0:

                    continue
                result[k][ts[start]] = result[k].get(ts[start],[])
                result[k][ts[start]] = v[start:end]
                result[k][ts[start]][0,2] = float(0.0)

        # Return result
        return result


def run(files, out_dir, ip_hosts):
    average_burst = 0
    print("number of files:",len(files))
    count_dic = {}
    for f in files:
        # parse pcap filename
        dir_name = os.path.dirname(f)
        activity = os.path.basename(dir_name)
        if activity[-1] == '2':
            activity = activity[:-1]
        dev_name = os.path.basename(os.path.dirname(dir_name))
        dir_target = os.path.join(out_dir, dev_name, activity)
        if not os.path.isdir(dir_target):
            os.system("mkdir -pv %s" % dir_target)
        if dev_name not in count_dic:
            count_dic[dev_name] = {'prior':0,'after':0,'whois':0,'blank':0,'local':0} 
        out_txt = os.path.join(dir_target, os.path.basename(f)[:-4] + "txt")
        #nothing happens if output file exists
        # print(dev_name, os.path.basename(f))
        if os.path.isfile(out_txt):
            print("%s exists" % out_txt)
        else:
            count_tmp = extract_pcap(f, out_txt, dev_name, ip_hosts[dev_name])
            count_dic[dev_name]['prior'] += count_tmp['prior']
            count_dic[dev_name]['after'] += count_tmp['after']
            count_dic[dev_name]['whois'] += count_tmp['whois']
            count_dic[dev_name]['blank'] += count_tmp['blank']
            count_dic[dev_name]['local'] += count_tmp['local']
    if not os.path.isdir('./logs/decode_logs'):
        os.system("mkdir -pv %s" % './logs/decode_logs')
    log_dir = os.path.join('logs', 'decode_logs', os.path.basename(os.path.dirname(out_dir)))
    if not os.path.isdir(log_dir):
        os.system("mkdir -pv %s" % log_dir)
    print('Log dir: ',log_dir)
    for dev_name, c in count_dic.items():
        print(dev_name, c)
        with open('%s/%s.txt' %(log_dir, dev_name), "a+") as flog:
            for k, v in c.items(): 
                flog.write('%s:%s  ' % (k, v))
            flog.write('\n')
        
    # print("Average burst num:", average_burst/len(files))

def main():
    global mac_dic
    [ print_usage(0) for arg in sys.argv if arg in ("-h", "--help") ]

    print("Running %s..." % sys.argv[0])

    #error checking
    #check for 2 or 3 arguments
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(c.WRONG_NUM_ARGS % (2, (len(sys.argv) - 1)))
        print_usage(1)

    in_txt = sys.argv[1]
    out_dir = sys.argv[2]
    str_num_proc = sys.argv[3] if len(sys.argv) == 4 else "1"

    #check in_txt
    errors = False
    if not in_txt.endswith(".txt"):
        errors = True
        print(c.WRONG_EXT % ("Input text file", "text (.txt)", in_txt), file=sys.stderr)
    elif not os.path.isfile(in_txt):
        errors = True
        print(c.INVAL % ("Input text file", in_txt, "file"), file=sys.stderr)
    elif not os.access(in_txt, os.R_OK):
        errors = True
        print(c.NO_PERM % ("input text file", in_txt, "read"), file=sys.stderr)

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
    num_proc = 1
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

    mac_dic = utils.read_mac_address()

    print("Input file located in: %s\nOutput files placed in: %s\n" % (in_txt, out_dir))

    #create groups to run TShark with processes
    in_files = [ [] for _ in range(num_proc) ]
    dev_proc = {}
    #split pcaps into num_proc groups
    with open(in_txt, "r") as f:
        index = 0
        
        for pcap in f:
            pcap = pcap.strip()
            if not pcap.endswith(".pcap"):
                print(c.WRONG_EXT % ("Input pcaps", "pcap (.pcap)", pcap))
            elif not os.path.isfile(pcap):
                print(c.INVAL % ("Input pcap", pcap, "file"))
            elif not os.access(pcap, os.R_OK):
                print(c.NO_PERM % ("input pcap", pcap, "read"))
            else:
                dir_name = os.path.dirname(pcap)    
                dev_name = os.path.basename(os.path.dirname(dir_name))
                if dev_name not in dev_proc:
                    dev_proc[dev_name] = index
                    index += 1

    with open(in_txt, "r") as f:
        for pcap in f:
            pcap = pcap.strip()
            if not pcap.endswith(".pcap"):
                print(c.WRONG_EXT % ("Input pcaps", "pcap (.pcap)", pcap))
            elif not os.path.isfile(pcap):
                print(c.INVAL % ("Input pcap", pcap, "file"))
            elif not os.access(pcap, os.R_OK):
                print(c.NO_PERM % ("input pcap", pcap, "read"))
            else:
                dir_name = os.path.dirname(pcap)
                dev_name = os.path.basename(os.path.dirname(dir_name))
                # if dev_name !='switchbot-hub':
                #     continue
                index = int(dev_proc[dev_name])
                in_files[index % num_proc].append(pcap)
                # index += 1

    # load ip_host mapping files generated by s1_decode_dns_tls.py
    ip_hosts_all = {}
    if in_txt.endswith('routine-dataset.txt'):
        model_file = '/home/ubuntu/Behaviot/event_inference/ip_host/routines.model'
    else:
        model_file = '/home/ubuntu/Behaviot/event_inference/ip_host/activity_nov.model'
    ip_hosts_all = pickle.load(open(model_file, 'rb'))
    
    procs = []
    for files in in_files:
        p = Process(target=run, args=(files, out_dir, ip_hosts_all))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

if __name__ == "__main__":
    main()

