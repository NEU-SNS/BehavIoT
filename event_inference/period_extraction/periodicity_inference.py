import pandas as pd
import numpy as np 
import os
import collections
import matplotlib
from statsmodels import api as sm
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 500
import matplotlib.pyplot as plt




# input files
root_feature = '../data/idle-2021-features/'
device_names = []
lparas = []
for csv_file in os.listdir(root_feature):
        if csv_file.endswith('.csv'):
            device_name = csv_file.replace('csv', '')
            device_names.append(device_name)
            train_data_file = os.path.join(root_feature, csv_file)


            dname = csv_file[:-4]


            lparas.append((train_data_file, dname))

lparas_sorted = sorted(lparas,key=lambda x:x[1])
for v in lparas_sorted:
    print(v[1])    

# load files
file_path = 'freq_period/2021_1s'   # output file
for a, b in enumerate(lparas):
    dname = lparas[a][-1]
    if os.path.isfile('%s/%s.txt' % (file_path, dname)):
        print('%sn exist' % dname)
        continue
    data = pd.read_csv(lparas[a][0])

    # set start and end time
    print('Dname ', dname)
    print(data.size)
    if dname=='govee-led1' or dname=='philips-bulb':    # experiment time differs from others 
        pass
    else:
        data = data.loc[(data['start_time'] <= 1631120400)] 
        data = data.loc[(data['start_time'] >= 1630688400)] 

    # get data
    nums = data['network_total'].values
    times = data['start_time'].values
    protocols = data['protocol'].values
    hosts = data['hosts'].fillna('').values
    X_feature = data.drop(['device', 'state', 'event','start_time','protocol', 'hosts'], axis=1).fillna(-1)
    if len(times) == 0:
        continue
    
    # preprocessing, optional
    preprocessing = 1
    if preprocessing:
        for i in range(len(protocols)):
            if 'TCP' in protocols[i]:
                protocols[i] = 'TCP'
            elif 'UDP' in protocols[i]:
                protocols[i] = 'UDP'
            elif 'TLS' in protocols[i]:
                protocols[i] = 'TLS'
            if ';' in protocols[i]:
                tmp = protocols[i].split(';')
                protocols[i] = ' & '.join(tmp)
                # print(protocols[i])
        protocol_set = set(protocols)
        print(protocol_set)
        
        for i in range(len(hosts)):
            if hosts[i] != '' and hosts[i] != None:
                tmp = hosts[i].split(';')
                hosts[i] = tmp[0]
            if hosts[i] == None:
                hosts[i] == 'non'
            hosts[i] = hosts[i].lower()
                # print(hosts[i])
        domain_set = set(hosts)
        print(domain_set)


    """
    Set Sampling Rate. In IMC23 paper, the sampling rate is set as 1 and 7200
    """
    sampling_rate = 1 # second
    binary = True # True: not consider the volumn of the flows 
    if sampling_rate!= 1:
        times = list(map(lambda x:round(x/sampling_rate), times)) # sampling rate
    times = list(map(int,times))
    max_time = np.max(times)
    min_time = np.min(times)
    print(max_time,min_time)


    """
    Iterate each protocol and domain pair 
    """
    for cur_protocol in protocol_set:
        # if cur_protocol != 'TCP':
        #     continue
        cur_domain_set = set()
        for i in range(len(times)):
            if protocols[i]== cur_protocol:
                cur_domain_set.add(hosts[i])

        """
        merge domain names with the same suffix
        """

        for i in cur_domain_set.copy():
            matched = 0
            if len(i.split('.')) >= 4:
                suffix = '.'.join([i.split('.')[-3], i.split('.')[-2], i.split('.')[-1]])
                for j in cur_domain_set.copy():
                    if j == i or j.startswith('*'):
                        continue
                    elif j.endswith(suffix):
                        matched = 1
                        
                        cur_domain_set.remove(j)
                        print('Remove : ',j)
                        print(cur_domain_set)
                if matched == 1:
                    cur_domain_set.remove(i)
                    print('Remove : ',i)
                    cur_domain_set.add('*.'+suffix)
        

        print('Protocol %s, domain set:' % cur_protocol)
        print(cur_domain_set)

        for cur_domain in cur_domain_set:

            print('Protocol %s, domain: %s' % (cur_protocol,cur_domain))

            domain_count = {}
            count_dic ={}
            cur_feature = []
            filter_feature = []
            for i in range(len(times)):
                if cur_domain.startswith('*'):
                    matched_suffix = hosts[i].endswith(cur_domain[2:])
                else:
                    matched_suffix = False
                if protocols[i]== cur_protocol and (matched_suffix or hosts[i] == cur_domain) : #  
                    if cur_domain in domain_count:
                        domain_count[cur_domain] += 1
                    else:
                        domain_count[cur_domain] = 1
                    
                    # if protocols[i]== 'GQUIC':
                    if times[i] in count_dic:
                        if binary:
                            count_dic[times[i]] += 1
                        else:
                            count_dic[times[i]] += nums[i]
                    else:
                        if binary:
                            count_dic[times[i]] = 1
                        else:
                            count_dic[times[i]] = nums[i]
                        

                    filter_feature.append(True)
                else:
                    filter_feature.append(False)
            cur_feature = X_feature[filter_feature]
            
            print('Domain count flow:', domain_count)
            domain_count2 = len(count_dic.keys())

            print('Domain count unique block:', domain_count2)
            if count_dic == {}:
                continue
            # if domain_count[cur_domain] <= 10:
            #     continue
            
            '''
            min time = start time
            '''
            min_time_tmp = min_time
            # min_time_tmp = np.min(list(count_dic.keys()))
            while(min_time_tmp <= max_time):
                if min_time_tmp not in count_dic:
                    count_dic[min_time_tmp] = 0
                min_time_tmp += 1 

            requestOrdered = dict(collections.OrderedDict(sorted(count_dic.items(), key=lambda t: t[0])))
            x = list(requestOrdered.keys())
            x_min_tmp = x[0]
            x = list(map(lambda x:x-x_min_tmp,x))
            y = list(requestOrdered.values())


            os.makedirs('%s/%s' % (file_path,dname), exist_ok=True)

            """
            Plot time domain 
            """
            plt.figure()
            plt.plot(x, y) #
            plt.grid()
            plt.xlabel('time')
            plt.ylabel('volume')
            plt.yscale("log")
            plt.title('%s'% (dname))
            plt.savefig('%s/%s/%s_%s.png' % (file_path,dname,cur_domain, cur_protocol))
            # plt.show()
            plt.close()
            count=0
            time_list =  []
            if domain_count2 < 30:
                for i in y:
                    if i > 0:
                        time_list.append(count)
                    count+=1

            """
            Frequency analysis
            """
            # N = 800
            N = len(x)  # number of signal points
            print('N:', N)
            # sample spacing
            # T = 1.0 / ( N) 
            T = N / N
            # sampling frequency 
            f_s = 1/T
            
            yf = fft(y)
            xf = fftfreq(N, T)[:N//2]
            tmp_max = np.max(np.abs(yf[0:N//2]))
            tmp_mean = np.mean(np.abs(yf[0:N//2]))
            tmp_std = np.std(np.abs(yf[0:N//2]))
            print(tmp_max, tmp_mean,tmp_std)
            
            """
            We use DFT to extract candidate periods for a group by identifying the frequencies that carry significant power in spectral density. The period candidates are the inverse of significant frequencies. To discover the significance in spectral density, a threshold need to be set. We apply an approach from prior work to generate random permutations of the input sequence that do not have any periodicity. The maximum power of each permutation can be used as an indicator of aperiodicity.
            """ 
            """
            @inproceedings{li2010mining,
            title={Mining periodic behaviors for moving objects},
            author={Li, Zhenhui and Ding, Bolin and Han, Jiawei and Kays, Roland and Nye, Peter},
            booktitle={Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining},
            pages={1099--1108},
            year={2010}
            }
            @inproceedings{vlachos2005periodicity,
            title={On periodicity detection and structural periodic similarity},
            author={Vlachos, Michail and Yu, Philip and Castelli, Vittorio},
            booktitle={Proceedings of the 2005 SIAM international conference on data mining},
            pages={449--460},
            year={2005},
            organization={SIAM}
            }
            """
            # permutation 100 times to set threshold
            p_max_list = []
            for i in range(100):
                y_shuffle = np.random.permutation(y).tolist()
                p_max_list.append(np.max(np.abs(fft(y_shuffle)[1:N//2]).tolist()))
            threshold_99 = sorted(p_max_list)[-6]
            if sampling_rate >= 600:
                threshold_99 = sorted(p_max_list)[-11]
                print('Threshold 90 percentile: ', threshold_99)
            else:
                print('Threshold 95 percentile: ', threshold_99)
            
            
            tmp_list = []
            tmp_list_yf = []
            for i in range(len(yf[0:N//2])):
                if i == 0 or i == 1  or i ==len(yf)-1: # or i < N/10000
                    continue
                if np.abs(yf[i]) > threshold_99:
                    tmp_list.append(i)
                    tmp_list_yf.append(np.abs(yf[i]))

            print('List len > threshold:', len(tmp_list))
            print('yf index: ', tmp_list[:10])
            print('yf: ', tmp_list_yf[:10])
            
            print('zipped:', sorted(list(zip(tmp_list,tmp_list_yf)), key = lambda x:x[1], reverse = True)[:5])
        
            period = []
            period_tmp_list = []
            if len(tmp_list) >0:
                for i in range(len(tmp_list)):
                    if sampling_rate >600 or round(N/tmp_list[i]) >= 10:
                        if len(period) == 0:
                            period.append(round(N/tmp_list[i]))
                            period_tmp_list.append(tmp_list[i])
                        else:
                            if round(N/tmp_list[i]) != period[-1]:
                                period.append(round(N/tmp_list[i]))
                                period_tmp_list.append(tmp_list[i])
            
            print('Protocol %s, domain: %s' % (cur_protocol,cur_domain))
            print('period:',period[:])
            
            """
            Then, we use autocorrelation to validate the period candidates and identify the true period for each pattern. The periods that have a significant autocorrelation score are chosen as the final periods of the signal. 
            """
            acf = sm.tsa.acf(y, nlags=len(y),fft=True)
        
            autocorrelation = []
            if len(period) == 0:
                pass
                # period.append(60)
            else:
                for i in range(len(period)):
                    tmp_range = [max(round(N/(period_tmp_list[i]-1)),period[i]+1), min(round(N/(period_tmp_list[i]+1)),period[i]-1)]

                    j = tmp_range[0]
                    while (j >= tmp_range[1]):
                        if j >= len(acf):
                            break
                        auto_tmp = acf[j]
                        if auto_tmp >= 3.315/np.sqrt(N):
                            autocorrelation.append(((j,auto_tmp))) # '%d:%d ' % 
                        j-=1
                autocorrelation = set(autocorrelation)
                autocorrelation = sorted(autocorrelation,key=lambda x:x[1], reverse = True)
                if len(autocorrelation) > 20:
                    print('## Autocorrelation ',autocorrelation[:20])
                else:
                    print('## Autocorrelation ',autocorrelation)
            
            # special case that has only few data points: 
            if not any(autocorrelation) and domain_count2 <= 6 and domain_count2 >= 4:
                # autocorrelation = []
                time_diff = [abs(time_list[i + 1] - time_list[i]) for i in range(len(time_list)-1)]
                diff_diff = [abs(time_diff[i + 1] - time_diff[i]) for i in range(len(time_diff)-1)]
                res = [x for x in diff_diff if x <= 3600/sampling_rate]
                if len(res)==len(diff_diff):
                    autocorrelation.append((np.mean(time_diff),0))
                    print('## Less than 6 data points: period ',autocorrelation)
                # print(time_diff)


            """
            Plot frequency domain
            """
            plt.figure()
            plt.plot(xf, 1.0/N * np.abs(yf[0:N//2]))
            plt.grid()
            plt.savefig('%s/%s/%s_%s_fft.png' % (file_path,dname,cur_domain, cur_protocol))
            plt.close()

            print('--------------------------------------------------------')
            
            with open('%s/%s.txt' % (file_path,dname), 'a+') as file:
                    if len(period) > 0 and any(autocorrelation): # and len(acf_burst) > 1
                        file.write('\n%s %s # %d: ' %(cur_protocol,cur_domain,domain_count[cur_domain]))
                        file.write(' best: %d'% (list(autocorrelation)[0][0]  ))
                        if len(list(autocorrelation)) > 1:
                            file.write(', %d'% (list(autocorrelation)[1][0]  ))
                        
                
                    else:
                        file.write('\nNo period detected %s %s # %d ' %(cur_protocol,cur_domain, domain_count[cur_domain])) 
    
