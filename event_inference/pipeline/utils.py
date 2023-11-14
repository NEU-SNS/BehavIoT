import numpy as np
from subprocess import Popen, PIPE
import ipaddress

def validate_ip_address(address):
    """ check if it's a valid ip address

    Args:
        address (string): ip address

    Returns:
        bool: true as valid 
    """
    try:
        ip = ipaddress.ip_address(address)
        # print("IP address {} is valid. The object returned is {}".format(address, ip))
        return True
    except ValueError:
        # print("IP address {} is not valid".format(address)) 
        return False

def is_local(ip_src, ip_dst): 
    """ check if it's a local ip address 

    Args:
        ip_src (string): ip_scr
        ip_dst (string): ip_dst

    Returns:
        bool: true for yes
    """
    is_local = False
    try:
        is_local = (ipaddress.ip_address(ip_src).is_private and ipaddress.ip_address(ip_dst).is_private
                ) or (ipaddress.ip_address(ip_src).is_private and (ip_dst=="129.10.227.248" or ip_dst=="129.10.227.207")
                ) or (ipaddress.ip_address(ip_dst).is_private and (ip_src=="129.10.227.248" or ip_src=="129.10.227.207"))
    except:
        # print('Error:', ip_src, ip_dst)
        return 1
    return is_local

def dig_x(ip):
    domain_name = ''
    # dig - x ip +short
    command = ["dig", "-x", ip, "+short"]
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    # Get output. Give warning message if any
    out, err = process.communicate()
    # print(out.decode('utf-8').split('\n'))
    return out.decode('utf-8').split('\n')[0]


def protocol_transform(test_protocols):
    for i in range(len(test_protocols)):
        if 'TCP' in test_protocols[i]:
            test_protocols[i] = 'TCP'
        elif 'MQTT' in test_protocols[i]:
            test_protocols[i] = 'TCP'
        elif 'UDP' in test_protocols[i]:
            test_protocols[i] = 'UDP'
        elif 'TLS' in test_protocols[i]:
            test_protocols[i] = 'TCP'
        if ';' in test_protocols[i]:
            tmp = test_protocols[i].split(';')
            test_protocols[i] = ' & '.join(tmp)
    return test_protocols


def read_mac_address():
    mac_file = '../../devices.txt'
    mac_dic = {}
    with open(mac_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp_mac, tmp_device = line[:-1].split(' ')
            if len(tmp_mac) != 17:
                mac_split = tmp_mac.split(':')
                for i in range(len(mac_split)):
                    if len(mac_split[i]) != 2:
                        mac_split[i]='0'+mac_split[i]
                tmp_mac = ':'.join(mac_split)
            mac_dic[tmp_device] = tmp_mac
    # print(mac_dic)
    return mac_dic


def get_features():

    # cols_feat = [ "meanBytes", "minBytes", "maxBytes", "medAbsDev",
    #          "skewLength", "kurtosisLength", "meanTBP", "varTBP", "medianTBP", "kurtosisTBP",
    #          "skewTBP", "network_total", "network_in", "network_out", "network_external", "network_local",
    #         "network_in_local", "network_out_local","meanBytes_out_external",
    #         "meanBytes_in_external", "meanBytes_out_local", "meanBytes_in_local", "device", "state", "event", "start_time","protocol","hosts"]
    cols_feat = [ "meanBytes", "minBytes", "maxBytes", "medAbsDev",
             "skewLength", "kurtosisLength", "meanTBP", "varTBP", "medianTBP", "kurtosisTBP",
             "skewTBP", "network_total", "network_in", "network_out", "network_external", "network_local",
            "network_in_local", "network_out_local", "meanBytes_out_external",
            "meanBytes_in_external", "meanBytes_out_local", "meanBytes_in_local", 
            "device", "state", "event", "start_time", "remote_ip", "remote_port" ,"trans_protocol", "raw_protocol", "protocol", "hosts"]
    return cols_feat



def label_aggregate(y_labels, y_test, dname):
    """
    aggregate indistinguishable labels
    """

    on_off_list = ['echoplus', 'echospot','echodot3c','echodot4a', 'insteon-hub', 
        'lightify-hub',  'nest-tstat', 'magichome-strip', 'philips-bulb', 'smartthings-hub',
        'xiaomi-hub','xiaomi-strip', 'govee-led1','switchbot-hub','gosund-bulb1','bulb1',
        'smartlife-bulb', 'ikea-hub','t-wemo-plug', 'tplink-bulb']

    smart_speaker_list = ['echospot']
    all_camera_list = ['microseven-camera', 'icsee-doorbell',
        'tuya-camera', 'yi-camera', 'lefun-cam-wired', 'wyze-cam', 'luohe-spycam', 'amcrest-cam-wired', 
    'wansview-cam-wired' , 'dlink-camera', 'ubell-doorbell']
    ring_camera_list = ['ring-camera', 'ring-doorbell']
    color_dim_list = ['magichome-strip'] 
    tstat_list = ['nest-tstat']


    if dname in smart_speaker_list:
        for i in range(len(y_labels)):
            if (y_labels[i] == 'android_wan_volume' or y_labels[i] == 'android_lan_volume' ) : # and num_lables_1 > 2
                y_labels[i] = 'on_off'
        for i in range(len(y_labels)):
            if (y_labels[i] == 'local_voice' or y_labels[i] == 'local_volume' ) : # and num_lables_1 > 2
                y_labels[i] = 'local_voice'

    for i in range(len(y_labels)):
        if y_labels[i].startswith('alexa_') or y_labels[i].startswith('android_lan_') \
          or y_labels[i].startswith('android_wan_') or y_labels[i].startswith('local_'): # alexa_ android_lan_ android_wan_
            y_labels[i] = y_labels[i].split('_')[-1]
    
    
    

    if dname in on_off_list:
        num_lables_1 = len(set(y_labels))
        for i in range(len(y_labels)):
            if (y_labels[i] == 'on' or y_labels[i] == 'off' ) : # and num_lables_1 > 2
                y_labels[i] = 'on_off'

        num_lables_2 = len(set(y_labels))
        for i in range(len(y_labels)):
            if (y_labels[i] == 'color' or y_labels[i] == 'dim' ) and num_lables_2 > 2:
                y_labels[i] = 'color_dim'

        num_lables_3 = len(set(y_labels))
        for i in range(len(y_labels)):
            if (y_labels[i] == 'lock' or y_labels[i] == 'unlock' ) and num_lables_3 > 2:
                y_labels[i] = 'lock_unlock'

    if dname in all_camera_list:
        for i in range(len(y_labels)):
            if (y_labels[i] == 'audio' or y_labels[i] == 'photo' or y_labels[i] == 'recording' or y_labels[i] == 'watch' ) : # and num_lables_1 > 2
                y_labels[i] = 'audio_photo_recording_watch'
    if dname in ring_camera_list:
        for i in range(len(y_labels)):
            if (y_labels[i] == 'audio' or y_labels[i] == 'stop' or  y_labels[i] == 'watch' ) : # and num_lables_1 > 2
                y_labels[i] = 'audio_stop_watch_motion'
    
    if dname in color_dim_list:
        for i in range(len(y_labels)):
            if (y_labels[i] == 'color_dim' or y_labels[i] == 'on_off' or y_labels[i] == 'scene' ):
                y_labels[i] = 'toggle'
    if dname in tstat_list:
        for i in range(len(y_labels)):
            if (y_labels[i] == 'set' or y_labels[i] == 'temp' ):
                y_labels[i] = 'set'

    if dname in smart_speaker_list:
        for i in range(len(y_test)):
            if (y_test[i] == 'android_wan_volume' or y_test[i] == 'android_lan_volume' ) : # and num_lables_1 > 2
                y_test[i] = 'on_off'
        for i in range(len(y_test)):
            if (y_test[i] == 'local_voice' or y_test[i] == 'local_volume' ) : # and num_lables_1 > 2
                y_test[i] = 'local_voice'

    for i in range(len(y_test)):
        if y_test[i].startswith('alexa_') or y_test[i].startswith('android_lan_') \
          or y_test[i].startswith('android_wan_') or y_test[i].startswith('local_'): # alexa_ android_lan_ android_wan_
            y_test[i] = y_test[i].split('_')[-1]
    if dname in on_off_list:
        for i in range(len(y_test)):
            if (y_test[i] == 'on' or y_test[i] == 'off' ) : # and num_lables_1 > 2
                y_test[i] = 'on_off' 
        for i in range(len(y_test)):
            if (y_test[i] == 'color' or y_test[i] == 'dim' ) and num_lables_2 > 2:
                y_test[i] = 'color_dim'
        for i in range(len(y_test)):
            if (y_test[i] == 'lock' or y_test[i] == 'unlock' ) and num_lables_3 > 2:
                y_test[i] = 'lock_unlock'
    if dname in all_camera_list:
        for i in range(len(y_test)):
            if (y_test[i] == 'audio' or y_test[i] == 'photo' or y_test[i] == 'recording' or y_test[i] == 'watch' ) :
                y_test[i] = 'audio_photo_recording_watch'
    if dname in ring_camera_list:
        for i in range(len(y_test)):
            if (y_test[i] == 'audio' or y_test[i] == 'stop' or  y_test[i] == 'watch' ) : # and num_lables_1 > 2
                y_test[i] = 'audio_stop_watch_motion'


    if dname in color_dim_list:
        for i in range(len(y_test)):
            if (y_test[i] == 'color_dim' or y_test[i] == 'on_off' or y_test[i] == 'scene' ) :
                y_test[i] = 'toggle'
    if dname in tstat_list:
        for i in range(len(y_test)):
            if (y_test[i] == 'set' or y_test[i] == 'temp' ):
                y_test[i] = 'set'

    return y_labels, y_test


def calculate_metrics(TN, FP, FN, TP):

    # true positive + false positive == 0, precision is undefined; When true positive + false negative == 0

    if TP == 0 or TP + FP == 0 or TP + FN == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        # False positive rate
    FPR = FP / (FP + TN)
        # False negative rate
    FNR = FN / (TP + FN)

    return precision, recall, f1, FPR, FNR, TP, FP, FN
