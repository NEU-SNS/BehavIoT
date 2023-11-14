import os
from datetime import datetime
"""
Build the log files for Synoptic
"""
base_dir = './traces'
root_feature = '%s/log_uncontrolled' % base_dir
list1 = []
for csv_file in os.listdir(root_feature):
    dname = '-'.join(csv_file.split('.')[0].split('-')[1:]) # csv_file[9:-4]
    print(dname, csv_file)

    f = open("%s/%s" % (root_feature, csv_file))

    for line in f:
        if len(line) <= 1 or line.startswith(' ') : # or line.startswith('1') 
            continue
        else:
            time = ':'.join(line.split(':')[:-1])
            label = line.split(':')[-1]
            if label.endswith('\n'):
                label = label[:-1]
            try:
                o1 = datetime.strptime(time[:-1], '%m/%d/%Y, %H:%M:%S.%f')
            except:
                print('Error', dname, time, label)
                # exit(1)
            print(o1, label)

            list1.append(('%s-%s %s'  % (dname,label, str(o1)),o1))

# print(list1)
list1 = sorted(list1,key=lambda x: x[1])


label = 0
time = 0
with open('%s/unctrl_feb' % base_dir, 'w') as off:
    
    traceID = 0
    first_time = 0

    last_time = 0
    # last_time_str = 0
    for i in list1:
        line = i[0]
        label = line.split()[0]
        d_str = ' '.join(line.split()[1:])
        # print(d_str)
        o = datetime.strptime(d_str, '%Y-%m-%d %H:%M:%S.%f')
        if traceID == 0:
            first_time = o.timestamp()
            off.write('---%s---\n' % d_str)
            traceID += 1
        new_time = o.timestamp() - first_time

        if (new_time > last_time + 60):
        #     traceID +=1 

            off.write('---%s---\n' % d_str)
            last_time = 0
            first_time = o.timestamp()
            new_time = 0
        else:    
            last_time = new_time
        
        off.write('%s, %s \n' % (label, new_time))

