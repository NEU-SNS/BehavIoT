import os
from datetime import datetime
import random
"""
Build the log files for Synoptic
"""
base_dir = './traces'
root_feature = '%s/log_routines' % base_dir
list1 = []
for csv_file in os.listdir(root_feature):

    dname = '-'.join(csv_file.split('.')[0].split('-')[1:]) # csv_file[9:-4]
    print(dname, csv_file)
    f = open("%s/%s" % (root_feature, csv_file))

    for line in f:
        if len(line) <= 1 or line.startswith(' ') or line.startswith('0') :
            continue
        else:
            time = ':'.join(line.split(':')[:-1])
            label = line.split(':')[-1]
            if label.endswith('\n'):
                label = label[:-1]
            if label =='unknown':
                continue
            try:
                o1 = datetime.strptime(time[:-1], '%m/%d/%Y, %H:%M:%S.%f')
            except:
                print('Error', dname, time, label)
                # exit(1)
            print(o1, label)
            list1.append(('%s-%s %s'  % (dname,label, str(o1)),o1))

# print(list1)
list1 = sorted(list1,key=lambda x: x[1])

# print(list1)
# exit(1)
'''
label = 0
time = 0
with open('%s/trace_log_may1' % base_dir, 'w') as off:
    
    traceID = 0
    first_time = 0

    last_time = 0
    for i in list1:
        line = i[0]
        label = line.split()[0]
        d_str = ' '.join(line.split()[1:])
        # print(d_str)
        o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
        if traceID == 0:
            first_time = o.timestamp()
            traceID += 1
            off.write('---%s---\n' % d_str)
        new_time = o.timestamp() - first_time

        if (new_time > last_time + 60):
        #     traceID +=1 

            off.write('---%s---\n' % d_str)
            last_time = 0
            first_time = o.timestamp()
            new_time = 0
        else:    
            last_time = new_time
        
        # new_time = float(o) - last_time

        off.write('%s, %s \n' % (label, new_time))
        # off.write('%s \n' % (','.join(['%.3f' %(new_time), (new_line_1+'-'+new_line_2)])))
'''

trace_list = []
with open('%s/trace_may1' % base_dir, 'w') as off:
    
    traceID = 0
    first_time = 0

    last_time = 0
    tmp_trace_list = []
    for i in list1:
        line = i[0]
        label = line.split()[0]
        d_str = ' '.join(line.split()[1:])
        # print(d_str)
        o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
        if traceID == 0:
            first_time = o.timestamp()
            traceID += 1
            off.write('------\n' )
        new_time = o.timestamp() - first_time

        if (new_time > last_time + 60):
        #     traceID +=1 
            trace_list.append(tmp_trace_list)
            tmp_trace_list = []
            off.write('------\n' )
            last_time = 0
            first_time = o.timestamp()
            new_time = 0
        else:    
            last_time = new_time
        tmp_trace_list.append('%s, %s \n' % (label, new_time))
        ### new_time = float(o) - last_time

        off.write('%s, %s \n' % (label, new_time))
        ### off.write('%s \n' % (','.join(['%.3f' %(new_time), (new_line_1+'-'+new_line_2)])))

split_time = '10/25/2021, 00:00:00.000005'
list_train = [] 
list_test = []
for x in list1:
    if x[1] <= datetime.strptime(split_time, '%m/%d/%Y, %H:%M:%S.%f'):
        list_train.append(x)
    else:
        list_test.append(x)
"""
## 5 fold split by time
split_time = ['10/21/2021, 00:00:00.000005', '10/22/2021, 00:00:00.000005','10/23/2021, 00:00:00.000005','10/25/2021, 00:00:00.000005', '10/26/2021, 00:00:00.000005']
last_split_time_list = ['10/20/2021, 00:00:00.000005','10/21/2021, 00:00:00.000005', '10/22/2021, 00:00:00.000005','10/23/2021, 00:00:00.000005','10/25/2021, 00:00:00.000005' ]
list_split = []
list_split_train = []
for i in range(len(split_time)):
    list_split.append([])
    list_split_train.append([])


for i in range(len(split_time)):
    last_split_time = last_split_time_list[i]
    cur_split_time = split_time[i]
    print(last_split_time, cur_split_time)
    for x in list1:
        
        if x[1] >= datetime.strptime(last_split_time, '%m/%d/%Y, %H:%M:%S.%f') and x[1] <= datetime.strptime(cur_split_time, '%m/%d/%Y, %H:%M:%S.%f'):
            list_split[i].append(x)
        else:
            list_split_train[i].append(x)

# ### 5 fold split by random num
# list_split = []
# list_split_train = []
# # s = list(range(len(list1)))
# # print(trace_list)
# # trace_list = trace_list + trace_list 
# random.shuffle(trace_list)
# s = [trace_list[i::5] for i in range(5)]
# for i in range(5):
#     list_split.append([])
#     list_split_train.append([])
# for i in range(len(s)):
#     list_split[i] = s[i]
#     for j in range(len(s)):
#         if i != j:
#             list_split_train[i] = list_split_train[i] + s[j]



print('Length list_split: ', len(list_split), len(list_split[0]))

base_dir = './traces'
# exit(1)
for cur_fold in range(len(list_split)):
    test_fingerprint = []
    with open('%s/trace_5fold_%s' % (base_dir, str(cur_fold)), 'w') as off:
        traceID = 0
        first_time = 0

        last_time = 0
        for i in list_split[cur_fold]:
            
            
            # off.write('------\n' )
            # tmp_fingerprint = []
            # for line in i:
            #     off.write(line)
            #     tmp_fingerprint.append(line.split(',')[0])
            # test_fingerprint.append(''.join(tmp_fingerprint))

            line = i[0]
            label = line.split()[0]
            d_str = ' '.join(line.split()[1:])
            # print(d_str)
            o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
            if traceID == 0:
                first_time = o.timestamp()
                traceID += 1
                tmp_fingerprint = []
                off.write('------\n' )
            new_time = o.timestamp() - first_time

            if (new_time > last_time + 60):
            #     traceID +=1 
                test_fingerprint.append(';'.join(tmp_fingerprint))
                tmp_fingerprint = []
                off.write('------\n' )
                last_time = 0
                first_time = o.timestamp()
                new_time = 0
            else:    
                last_time = new_time
            
            
            tmp_fingerprint.append(label)
            off.write('%s, %s \n' % (label, new_time))
            

    train_fingerprint = []
    with open('%s/trace_5fold_train_%s' % (base_dir, str(cur_fold)), 'w') as off:
        traceID = 0
        first_time = 0

        last_time = 0
        for i in list_split_train[cur_fold]:
            # off.write('------\n' )
            # tmp_fingerprint = []
            # for line in i:
            #     off.write(line)
            #     tmp_fingerprint.append(line.split(',')[0])
            # train_fingerprint.append(''.join(tmp_fingerprint))

            line = i[0]
            label = line.split()[0]
            d_str = ' '.join(line.split()[1:])
            # print(d_str)
            o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
            if traceID == 0:
                first_time = o.timestamp()
                traceID += 1
                tmp_fingerprint = []
                off.write('------\n' )
            new_time = o.timestamp() - first_time

            if (new_time > last_time + 60):
            #     traceID +=1 
                train_fingerprint.append(';'.join(tmp_fingerprint))
                tmp_fingerprint = []
                off.write('------\n' )
                last_time = 0
                first_time = o.timestamp()
                new_time = 0
            else:    
                last_time = new_time
            
            
            tmp_fingerprint.append(label)
            off.write('%s, %s \n' % (label, new_time))
            

    count_new_trace = 0
    count_all = 0
    count_new_trace_transition = 0
    coutn_all_transition = 0
    for tmp_f in test_fingerprint:
        if tmp_f not in train_fingerprint:
            count_new_trace += 1
            count_new_trace_transition += len(tmp_f.split(';'))
        count_all += 1
    print('Count previously unseen trace:', cur_fold, count_new_trace, count_all, count_new_trace/count_all, count_new_trace_transition)
"""
dup_time = 50
with open('%s/trace_train_dup_may30' % base_dir, 'w') as off:
    
    for i in range(dup_time):

        traceID = 0
        first_time = 0

        last_time = 0
        for i in list_train:
            line = i[0]
            label = line.split()[0]
            d_str = ' '.join(line.split()[1:])
            # print(d_str)
            o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
            if traceID == 0:
                first_time = o.timestamp()
                traceID += 1
                off.write('------\n' )
            new_time = o.timestamp() - first_time

            if (new_time > last_time + 60):
            #     traceID +=1 

                off.write('------\n' )
                last_time = 0
                first_time = o.timestamp()
                new_time = 0
            else:    
                last_time = new_time
            
            # new_time = float(o) - last_time

            off.write('%s, %s \n' % (label, new_time))
            # off.write('%s \n' % (','.join(['%.3f' %(new_time), (new_line_1+'-'+new_line_2)])))


with open('%s/trace_train_log_dup_may30' % base_dir, 'w') as off:
    for i in range(dup_time):
        traceID = 0
        first_time = 0

        last_time = 0
        for i in list_train:
            line = i[0]
            label = line.split()[0]
            d_str = ' '.join(line.split()[1:])
            # print(d_str)
            o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
            if traceID == 0:
                first_time = o.timestamp()
                traceID += 1
                off.write('---%s---\n' % d_str)
            new_time = o.timestamp() - first_time

            if (new_time > last_time + 60):
            #     traceID +=1 

                off.write('---%s---\n' % d_str)
                last_time = 0
                first_time = o.timestamp()
                new_time = 0
            else:    
                last_time = new_time
            
            # new_time = float(o) - last_time

            off.write('%s, %s \n' % (label, new_time))
            # off.write('%s \n' % (','.join(['%.3f' %(new_time), (new_line_1+'-'+new_line_2)])))

with open('%s/trace_test_dup_may30' % base_dir, 'w') as off:
    for i in range(dup_time):
        traceID = 0
        first_time = 0

        last_time = 0
        for i in list_test:
            line = i[0]
            label = line.split()[0]
            d_str = ' '.join(line.split()[1:])
            # print(d_str)
            o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
            if traceID == 0:
                first_time = o.timestamp()
                traceID += 1
                off.write('---%s---\n' % d_str)
            new_time = o.timestamp() - first_time

            if (new_time > last_time + 60):
            #     traceID +=1 

                off.write('---%s---\n' % d_str)
                last_time = 0
                first_time = o.timestamp()
                new_time = 0
            else:    
                last_time = new_time
            
            # new_time = float(o) - last_time

            off.write('%s, %s \n' % (label, new_time))
            # off.write('%s \n' % (','.join(['%.3f' %(new_time), (new_line_1+'-'+new_line_2)])))
