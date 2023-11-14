import os
import sys
import networkx as nx
import pydot
import math
import numpy as np
from datetime import datetime
import scipy
import pickle
from scipy.stats import kurtosis, skew
import matplotlib 
from scipy.stats.mstats import gmean
import copy
import matplotlib.pyplot as plt
import collections
from collections import Counter
import time
matplotlib.use('Agg')


state_machine_file = sys.argv[1] + '.dot'
print(state_machine_file)
train_file = sys.argv[2]
print(train_file)
threshold_1 = 0
threshold_2 = 0 
threshold_3 = 0
decay_factor = 0
ci = 1.96


runtime_transition_dic = {}
class Monitor(object):
    # TODO: set and get parameters of PFSM
    def __init__(self):
        self.threshold = None
        self.decay_factor = None
    
    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_decay_factor(self, decay_factor):
        self.decay_factor = decay_factor
    
    def get_threshold(self):
    
        return self.threshold

    def get_decay_factor(self):

        return self.decay_factor

    threshold = property(set_threshold, get_threshold)
    decay_factor = property(set_decay_factor, get_decay_factor)

def plotting_cdf_list(score_list_list, name, file_list):
    plt.figure()
    if len(score_list_list)>8:
        n = len(score_list_list)
        color_list = plt.cm.rainbow(np.linspace(0, 1, n))
    else:
        color_list = ['r', 'b', 'y', 'g', 'c', 'k', 'm', 'w']
    
    for i in range(len(score_list_list)):
        score_list = score_list_list[i]
        cur_file = file_list[i]

        if 'train' in cur_file:
            cur_file = 'train'
        if 'test' in cur_file:
            cur_file = 'test'
        count_dic = Counter(score_list)
        # print(score_list)
        # print(count_dic)
        
        length_list = len(score_list)
        norm_factor = max(score_list)
        requestOrdered = dict(collections.OrderedDict(sorted(count_dic.items(), key=lambda t: t[0])))
        x = np.array(list(requestOrdered.keys()))#/norm_factor    # score 
        y = np.array(list(requestOrdered.values()))/length_list   # num of traces
        
        y = np.cumsum(y)
        plt.plot(x, y, 'r', label='%s' % cur_file.split('_')[-1], color=color_list[i]) # , 'o'
        if not name.startswith('transition'):
            if 'train' in cur_file:
                plt.text(x[round(len(x)/2)], y[round(len(y)/2)], cur_file.split('_')[-1])
            elif 'test' in cur_file:
                plt.text(x[round(len(x)/2)], y[round(len(y)/2)-2], cur_file.split('_')[-1])
            else:
                plt.text(x[round(len(x)/2)], y[round(len(y)/2)], cur_file.split('_')[-1])
                
        # formatter = FuncFormatter(to_percent)
        # plt.gca().yaxis.set_major_formatter(PercentFormatter())
        # urllength += len(test[i].urls)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.grid()
        
        # plt.xscale("log")
        # plt.title('%s'%name)
    # plt.legend()
    # plt.vlines(x=13.98, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(13.98, y[round(len(y)/2)], '$\sigma$')
    # plt.vlines(x=19.28, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(19.28, y[round(len(y)/2)], '$2\sigma$')
    # plt.vlines(x=24.58, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(24.58, y[round(len(y)/2)], '$3\sigma$')
    # plt.vlines(x=29.88, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(29.88, y[round(len(y)/2)], '$4\sigma$')
    # plt.vlines(x=35.18, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(35.18, y[round(len(y)/2)], '$5\sigma$')

    # plt.vlines(x=1.44, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(1.44, 0.48, '85%', fontsize='small')
    # plt.vlines(x=1.645, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(1.645, 0.52, '90%', fontsize='small')
    # plt.vlines(x=1.96, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(1.96, 0.5, '95%', fontsize='small')
    # plt.vlines(x=2.57, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(2.57, 0.48, '99%', fontsize='small')
    # plt.vlines(x=2.80, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.text(2.80, 0.52, '99.5%', fontsize='small')
    plt.xlabel('score')
    plt.ylabel('CDF')
    # if 'transition' in name:
    #     plt.ylabel('transitions')
    # else:
    #     plt.ylabel('traces')
    dic = './cdf/score_%s.pdf' % name
    plt.savefig(dic)
    dic = './cdf/score_%s.png' % name
    plt.savefig(dic)


    return 0

def plotting_cdf(score_list, name):

    count_dic = Counter(score_list)
    # print(score_list)
    # print(count_dic)
    plt.figure()
    length_list = len(score_list)
    norm_factor = max(score_list)
    requestOrdered = dict(collections.OrderedDict(sorted(count_dic.items(), key=lambda t: t[0])))
    x = np.array(list(requestOrdered.keys()))#/norm_factor    # score 
    y = np.array(list(requestOrdered.values()))/length_list   # num of traces
    
    y = np.cumsum(y)
    plt.plot(x, y, 'r') # , 'o'
    # formatter = FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter())
    # urllength += len(test[i].urls)
    # plt.yscale('log')
    # plt.xscale('log')
    if 'unctrl' in name:
        if 'transition' in name:
            plt.vlines(x=ci, ymin=0, ymax=1, color='k', linestyle='--')
        else:
            plt.vlines(x=19.28, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.grid()
    plt.tight_layout(2.0)

    plt.xlabel('Score', fontsize=15)
    plt.ylabel('CDF', fontsize=15)
    # plt.xscale("log")
    # plt.title('%s'%name)

    
    plt.subplots_adjust(left=0.1)
    plt.rcParams.update({'font.size': 15, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
    # plt.rc('font', size=15)  
    plt.rc('axes', labelsize=15)
    plt.rcParams.update({'legend.loc': 'lower right'})
    # plt.rc('axes', titlesize=15)
    # plt.ylim((0.99975,1.00001))
    # plt.vlines(x=1.65, ymin=0.99975, ymax=1, color='k', linestyle='--')
    # plt.vlines(x=0.68, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.vlines(x=0.63, ymin=0, ymax=1, color='k', linestyle='--')
    # plt.legend()
    dic = './cdf/score_%s.pdf' % name
    plt.savefig(dic)
    dic = './cdf/score_%s.png' % name
    plt.savefig(dic)


    return 0


def set_threshold(t1, t2, t3):
    global threshold_1, threshold_2, threshold_3
    threshold_1 = t1
    threshold_2 = t2
    threshold_3 = t3
    return 0

def set_decay_factor(d):
    global decay_factor
    decay_factor = d
    return 0

def get_threshold():
    # global threshold_1, threshold_2, threshold_3
    threshold = threshold_3
    return threshold

def get_decay_factor():
    # global decay_factor
    return decay_factor

def trace_forward_wrapper(last_node_id, trace, cur_pos_trace, mode):
    trace_id_list = []
    visited = set()
    new_transition_set = set()
    trace_id_list = trace_forward(last_node_id, trace, cur_pos_trace, mode, visited, trace_id_list, new_transition_set)
    return trace_id_list, new_transition_set

def num_of_visited(trace, largest_cur_pos_trace):
    global activity_node_dic
    num = []
    for i in range(len(trace)):
        if i <= largest_cur_pos_trace + 1:
            if trace[i] not in activity_node_dic:
                continue
            for x in activity_node_dic[trace[i]]:
                num.append(x)
    # print('num_of_visited: ', len(num))
    return len(num)

def trace_forward(last_node_id, trace, cur_pos_trace, mode, visited, trace_id_list, new_transition_set):
    global activity_node_dic, inverse_activity_node_dic, largest_cur_pos_trace, runtime_transition_dic, transition_dic, node_count_static, node_count
    event = trace[cur_pos_trace]
    # print(cur_pos_trace, event)
    if event not in activity_node_dic:
        
        max_node = 0
        for k,v in activity_node_dic.items():
            for m in v:
                if max_node < m: 
                    max_node = m
        print('New event: ', event, max_node+1)
        activity_node_dic[event] = [max_node+1]
        runtime_transition_dic[max_node+1] = {}
        transition_dic[max_node+1] = {}
        node_count[max_node+1] = 0
        node_count_static[max_node+1] = 0
        inverse_activity_node_dic[max_node+1] = event
        # trace_id_list.append(last_node_id)
        # return trace_id_list
    for i in activity_node_dic[event]:
        # print('Current node id:', i)
        visited.add((i,cur_pos_trace))
        if len(trace_id_list) > 0:
            # print('trace_id_list:',trace_id_list)
            trace_id_list.append(last_node_id)
            return trace_id_list
        if i in runtime_transition_dic[last_node_id]:
            
            # print('Transition:', last_node_id, i)
            if event == 'TERMINAL':
                trace_id_list = [i, last_node_id]
                # print('trace_id_list:',trace_id_list)
                return trace_id_list

            else:

                if cur_pos_trace > largest_cur_pos_trace:
                    largest_cur_pos_trace = cur_pos_trace 
                # print('largest_cur_pos_trace:', largest_cur_pos_trace)
                trace_id_list = trace_forward(i, trace, cur_pos_trace+1, mode, visited, trace_id_list, new_transition_set)
                # print('trace_id_list:',trace_id_list)
        else:
            # no transition from last_node_id to event 
            continue
    
    # print('trace_id_list:',trace_id_list)
    if len(trace_id_list) > 1:
        # print(trace_id_list)
        trace_id_list.append(last_node_id)
        return trace_id_list
    elif len(visited)==num_of_visited(trace, largest_cur_pos_trace):
        # no transition from last_node_id to all current event nodes. 
        last_event =  trace[largest_cur_pos_trace] if largest_cur_pos_trace!=-1 else 'INITIAL'
        # print('no transition from %s to %s  ' %(last_event, trace[largest_cur_pos_trace+1]))
        if mode == 1: # add mode
            if trace[largest_cur_pos_trace+1] not in activity_node_dic:
                return trace_id_list 
            runtime_transition_dic, tmp_tuple = add_new_edge(last_event, trace[largest_cur_pos_trace+1])
            new_transition_set.add(tmp_tuple)
            largest_cur_pos_trace = -1
            return trace_forward(last_node_id, trace, cur_pos_trace, mode, visited, [], new_transition_set)
        elif mode == 2:
            return trace_id_list 
    else:
        return trace_id_list


def is_valid_transition(start_event, end_event):
    global activity_node_dic
    s = activity_node_dic[start_event]
    e = activity_node_dic[end_event]
    
    return 0

def calculate_probability(*args):
    global runtime_transition_dic, node_count

    trace = args[0]
    tmp_new_transition_set = args[1]
    prob = 1
    last_node = trace[0]
    trace_length = len(trace)
    if trace_length == 0:
        return 0
    for i in trace[1:]:
        prob *= runtime_transition_dic[last_node][i][0]
        if (last_node,i) in tmp_new_transition_set:
            prob *= 0.5
        # print('%s to %s: %f'% (last_node, i, transition_dic[last_node][i][0]))
        last_node = i
    return prob


def add_new_edge(start_event,end_event):
    global runtime_transition_dic, node_count, activity_node_dic, inverse_activity_node_dic, transition_dic, new_edge_indicator
    e = activity_node_dic[end_event][-1]
    for s_tmp in activity_node_dic[start_event][::-1]:
        if e not in runtime_transition_dic[s_tmp]:
            s = s_tmp
            break
    node_count[s] += 1
    prob = float(1 / node_count[s])
    if e in runtime_transition_dic[s]:
        print('Already exists:', s, e, runtime_transition_dic[s])
        exit(1)
    runtime_transition_dic[s][e] = [prob, 1]
    transition_dic[s][e] = [prob, 1]
    print('Add new edge: ', s, e)
    new_edge_indicator = 1
    return runtime_transition_dic, (s,e) 

def get_transition_boundry():
    transition_boundry_dic = pickle.load(open("transition_boundry_dic.pkl","wb"))
    return transition_boundry_dic

# transition_dic is the original trained transition dic. 
# runtime_transition_dic is the runtime transition dic subject to change in runtime. 
# transition_boundry_dic: boundries. 

def check_edges(trace):
    global runtime_transition_dic, transition_boundry_dic, node_count, alert_set, transition_dic, transition_score_tmp, node_count_static

    # transition_boundry_dic = get_transition_boundry()
    last_node = trace[0]
    alert = 0
    res = 0
    trace_alert = 0
    cur_transition_score = []
    for i in trace[1:]:
        
        cur_prob = runtime_transition_dic[last_node][i][0]
        tmp_k = (last_node,i)
        
        if tmp_k in transition_boundry_dic:
            # print("In boundry dic: ",tmp_k)
            cur_prob = (runtime_transition_dic[last_node][i][1]+1) / (1 + node_count[last_node])
            
            p_0 = cur_prob
            # count = node_count[last_node]
            count = node_count_static[last_node]
            old_p_0 = transition_dic[last_node][i][0]
            try:
                if old_p_0 == 1.0:
                    old_p_0 = 0.999
                tmp_score = (cur_prob-old_p_0) / math.sqrt((old_p_0)*(1-old_p_0)/count)
            except:
                print('Error:', old_p_0, count, last_node, i)
                exit(1)
            if abs(tmp_score) > float(ci) and trace_alert==0:
                print('!!Score ',last_node, i,cur_prob, old_p_0,tmp_score)
            cur_transition_score.append(tmp_score)
                # trace_alert = 1
            if cur_prob <= transition_boundry_dic[(last_node,i)][0] and cur_prob >= transition_boundry_dic[(last_node,i)][1]: # good

                runtime_transition_dic[last_node][i][0] = (runtime_transition_dic[last_node][i][1]+1) / (1 + node_count[last_node])
                node_count[last_node] += 1
                # print('good:',last_node, i, runtime_transition_dic[last_node][i][0])
                runtime_transition_dic[last_node][i][1] += 1
                for k in runtime_transition_dic[last_node].keys():
                    if k != i:
                        runtime_transition_dic[last_node][k][0] = (runtime_transition_dic[last_node][k][1]) / (node_count[last_node])
            else:
                # print('Bad')
                alert_set.add((last_node,i))
                alert = 1
        else:
            runtime_transition_dic[last_node][i][0] = (runtime_transition_dic[last_node][i][1]+1) / (1 + node_count[last_node])
            node_count[last_node] += 1
            runtime_transition_dic[last_node][i][1] += 1
            for k in runtime_transition_dic[last_node].keys():
                if k != i:
                    runtime_transition_dic[last_node][k][0] = (runtime_transition_dic[last_node][k][1]) / (node_count[last_node])
            cur_prob = runtime_transition_dic[last_node][i][0]
            if node_count[last_node]*cur_prob >= 8 and node_count[last_node]*(1-cur_prob) >= 8:
                if inverse_activity_node_dic[i] == 'TERMINAL':
                    continue
                p_0 = cur_prob
                count = node_count[last_node]
                print('New transition boundry set: ', (last_node,i), (ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0))
                transition_boundry_dic[(last_node,i)] = [ ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0]
                transition_dic[last_node][i] = copy.deepcopy(runtime_transition_dic[last_node][i])
                node_count_static[last_node] = copy.deepcopy(node_count[last_node])
        
        if alert == 1:
            res = 1
            cur_prob = (runtime_transition_dic[last_node][i][1]+1) / (1 + node_count[last_node])
            if cur_prob > transition_boundry_dic[(last_node,i)][0]:
                print('Alert!!!! HIGH', last_node, i, cur_prob, transition_boundry_dic[(last_node,i)][0], tmp_score)

                runtime_transition_dic[last_node][i][0] = cur_prob
                node_count[last_node] += 1
                runtime_transition_dic[last_node][i][1] += 1
                for k in runtime_transition_dic[last_node].keys():
                    if k != i:
                        runtime_transition_dic[last_node][k][0] = (runtime_transition_dic[last_node][k][1]) / (node_count[last_node])
                p_0 = cur_prob
                transition_dic[last_node][i] = copy.deepcopy(runtime_transition_dic[last_node][i])
                node_count_static[last_node] = copy.deepcopy(node_count[last_node])
                count = node_count[last_node]
                transition_boundry_dic[(last_node,i)] = [ ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0]
                print('New boundry: ', ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0)
            else:
                print('Alert!!!! LOW', last_node, i, cur_prob, transition_boundry_dic[(last_node,i)][1],tmp_score)

                runtime_transition_dic[last_node][i][0] = cur_prob
                node_count[last_node] += 1
                runtime_transition_dic[last_node][i][1] += 1
                for k in runtime_transition_dic[last_node].keys():
                    if k != i:
                        runtime_transition_dic[last_node][k][0] = (runtime_transition_dic[last_node][k][1]) / (node_count[last_node])
                p_0 = cur_prob
                transition_dic[last_node][i] = copy.deepcopy(runtime_transition_dic[last_node][i])
                node_count_static[last_node] = copy.deepcopy(node_count[last_node])
                transition_boundry_dic[(last_node,i)] = [ ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0]

            alert = 0
        last_node = i
    return res, cur_transition_score


def testing(trace, trace_timestamp_id, update_boundry, trace_events, tmp_new_transition_set):
    """
    The main testing func
    """
    
    # Res1: trace_score, Res2: transition_score, Res3: invarient_score (rule-based)
    res1 = res2 =res3 = 0
    decay_factor = get_decay_factor()
    trace_length = len(trace)
    threshold = get_threshold()
    cur_prob = calculate_probability(trace, tmp_new_transition_set)
    cur_prob = cur_prob # * ((decay_factor)**(trace_length-3))
    cur_deviation_score = 1-cur_prob
    # threshold 2 and 3
    if cur_deviation_score > 1 or cur_prob == 0:
        print('Error score', cur_prob, trace)
        exit(1)
        
    cur_deviation_score = (1-math.log(cur_prob))
    if cur_prob == 0 or cur_deviation_score > threshold:
        res1 = 1
        print('Deviation score alert--', trace_timestamp_id, cur_deviation_score, trace)
    
    res2, cur_transition_score = check_edges(trace)
    res3 = invarient_testing(trace_events)
    if res3 == 1:
        print('Deviation score alert invarient--', trace_timestamp_id, trace)
    if update_boundry>=1:
        update_boundry_func(update_boundry)

    return (res1 or res2 or res3), cur_deviation_score, cur_transition_score


def update_boundry_func(update_boundry):
    """
    update transition probability dictionary, and decision boundry
    """
    global runtime_transition_dic, transition_boundry_dic, inverse_activity_node_dic,transition_dic, node_count, node_count_static
    print('Updating transition boundry dic --- ')
    new_transition_boundry_dic ={}
    
    if update_boundry==2:
        for k in runtime_transition_dic.keys():

            node_count[k] = node_count[k]/2
            # cut_off = node_count[k]/2
            for n in runtime_transition_dic[k].keys():
                runtime_transition_dic[k][n][1] = runtime_transition_dic[k][n][1]/2
                transition_dic[k][n][1] = copy.deepcopy(runtime_transition_dic[k][n][1])
                # node_count[n] = node_count[n] - runtime_transition_dic[k][n][1]/2

    for k, v in runtime_transition_dic.items():
        count = 0
        for e in v.keys():
            e_0 = e
            count += v[e][1]
            p_0 = v[e][0]
        # print(k,e_0, p_0, count)
       
        if node_count[k] != count:
            print('Not equal: ', node_count[k], count, k, v)
            exit(1)
        for e in v.keys(): 
            if inverse_activity_node_dic[e] == 'TERMINAL':
                continue
            e_0 = e
            p_0 = v[e][0]
            if (k,e) in transition_boundry_dic or (count*p_0 >= 6 and count*(1-p_0) >= 6 ):
                new_transition_boundry_dic[(k,e)] = [ ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0]
                transition_dic[k][e] = copy.deepcopy(runtime_transition_dic[k][e])
                node_count_static[k] = copy.deepcopy(node_count[k])
                if (k,e) not in transition_boundry_dic:
                    print('NEW Selected edge: ', k, e, v[e])
    transition_boundry_dic = new_transition_boundry_dic


def invarient_testing(trace):
    alwaysPrecedes = {'bulb1-on_off': 'echospot-voice' , 'govee-led1-color_dim': 'echospot-voice', 'amazon-plug-on': 'echospot-voice' ,'nest-tstat-set': 'echospot-voice' }
    alwaysFollowedBy = {'ring-camera-audio_stop_watch_motion':'gosund-bulb1-on_off','nest-tstat-set': 'meross-dooropener-close'}

    res = 0
    followedby_event = []
    for event in trace:
        if len(followedby_event)!=0:
            if event in followedby_event:
                followedby_event.remove(event)
        if event in alwaysFollowedBy.keys() and event not in followedby_event:
            followedby_event.append(alwaysFollowedBy[event])
    
    if len(followedby_event)!=0:
        res = 1
        print('Invarient violation, alwaysFollowedBy: ', followedby_event)
        remove_list = []
        for k,v in alwaysFollowedBy.items():
            if v in followedby_event:
                remove_list.append(k)
        
        for k in remove_list:
            alwaysFollowedBy.pop(k)

    precedes_event = []
    precedes_event_value = []
    trace_tmp = trace
    for i in range(len(trace)):
        event = trace[i]
        if event in alwaysPrecedes.keys():
            precedes_event = ['echospot-voice']
            precedes_event_value.append(event)
            for j in range(len(trace_tmp)):
                if j==i:
                    break
                if trace_tmp[j] in precedes_event:
                    precedes_event.remove(trace_tmp[j])
                    precedes_event_value.remove(event)
            break
    
    if len(precedes_event) != 0:
        res = 1
        
        print('Invarient violation, alwaysPrecedes: ', precedes_event)
        remove_list = []
        for k,v in alwaysPrecedes.items():
            if k in precedes_event_value:
                remove_list.append(k)
        for k in remove_list: 
            alwaysPrecedes.pop(k)
    return res


activity_node_dic = {} # key: activity, value: list of node number
transition_dic = {} # key: start node, value: (dic: key end nodes, value: [prob, count])



graphs = pydot.graph_from_dot_file(state_machine_file)
graph = graphs[0]
# DG = nx.DiGraph()
# DG = nx.DiGraph(read_dot(state_machine_file)) 
DG = nx.drawing.nx_pydot.from_pydot(graph)
# print(list(DG.nodes))
# print(DG.nodes.data())
# print(DG.nodes['0'])
for e in list(DG.nodes.data()):
    # print(e)
    node_num = int(e[0])
    label = e[1]['label'].replace('\"','')
    # print(node_num, label)
    if label not in activity_node_dic:
        activity_node_dic[label] = [node_num]
    else:
        activity_node_dic[label].append(node_num)

inverse_activity_node_dic = {}
for k, v in activity_node_dic.items():
    # print(k,v)
    for j in v:
        inverse_activity_node_dic[j] = k

"""
Process graph data: transition probability
"""

for s,e, dic in DG.edges.data():
    s = int(s)
    e = int(e)
    # print(s,e,dic)
    prob = float(dic['label'].replace('\"','').split(',')[0].split(':')[1])
    
    # count = round(node_count[inverse_activity_node_dic[s]] * prob)
    count = int(dic['label'].replace('\"','').split(',')[1].split(':')[1])
    
    # print(prob, count)
    if s not in transition_dic:
        transition_dic[s] = {e:[prob, count]}
    else:
        transition_dic[s][e] = [prob, count]

node_count = {} 
for k, v in transition_dic.items():
    counter = 0
    # print(k, v)
    for i in v.keys():
        counter += v[i][1]
    node_count[k] = counter

for k, v in transition_dic.items():
    for e in v.keys():
        prob = transition_dic[k][e][0]
        count = transition_dic[k][e][1]
        prob2 = float(count/node_count[k])
        transition_dic[k][e][0] = prob2
        if node_count[k] != count/prob2:
            print(prob, node_count[k], count/prob2)

node_count_static = copy.deepcopy(node_count)
# exit(0)
transition_boundry_dic = {}
for k, v in transition_dic.items():
    print('---')
    print(k,v)
    e_0 = 0
    count = 0
    for e in v.keys():
        e_0 = e
        count += v[e][1]
        p_0 = v[e][0]
    # print(k,e_0, p_0, count)
    for e in v.keys(): 
        if inverse_activity_node_dic[e] == 'TERMINAL':
            continue
        e_0 = e
        p_0 = v[e][0]
        if (count*p_0 >= 6 and count*(1-p_0) >= 6 ) or (p_0>=0.9 and count>=3):
            print('Selected edge: ', k, e, v[e])
            print('Boundry: ', ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0)
            transition_boundry_dic[(k,e)] = [ ci*math.sqrt((p_0)*(1-p_0)/count)+p_0, -ci*math.sqrt((p_0)*(1-p_0)/count)+p_0]

runtime_transition_dic = copy.deepcopy(transition_dic)


"""
Threshold calculating for short-term deviation metric
"""

train_file = train_file
with open(train_file, 'r') as f:
    lines = f.readlines()
    prob = 1
    last_node = ''
    mean_prob = []
    mean_prob_geometric = []
    trace_id = 0
    trace_list = []
    line_tmp = []
    # read the file
    for line in lines:
        line = line.strip()
        
        if line.startswith('---'):
            if trace_id != 0:
                line_tmp.append('TERMINAL')
                trace_list.append(line_tmp)
            trace_id += 1
            line_tmp = []
        else:
            line_tmp.append(line.split(',')[0])
    line_tmp.append('TERMINAL')
    trace_list.append(line_tmp)
    print('Trace length:', trace_id)
    # print(trace_list)
    prob_per_len = {}
    baseline_length = []

    
    # Old calculate without decay
    # for each trace 
    for trace in trace_list: # chronological order
        last_node = activity_node_dic['INITIAL']
        # trace_id_list = []
        prob = 1
        trace_length = len(trace)
        # if trace_length == 12:
        #     print(trace)
        baseline_length.append(trace_length)
        cur_pos_trace = 0
        largest_cur_pos_trace = -1
        print(trace)
        trace_id_list, tmp_new_transition_set = trace_forward_wrapper(last_node[0], trace, cur_pos_trace, 1)
        print(trace_id_list[::-1])
        if not isinstance(trace_id_list, list) or len(trace_id_list) <= 1:
            print('No transition from %s to %s  ' %(trace[largest_cur_pos_trace], trace[largest_cur_pos_trace+1]))
            continue

        # trace_id_list is a list of node id accoiated with the events
        prob = calculate_probability(trace_id_list[::-1], tmp_new_transition_set)
        if trace_length not in prob_per_len:
            prob_per_len[trace_length] = [prob]
        else:
            prob_per_len[trace_length].append(prob)

            
        if prob == 0:
            pass
            # print('Prob = 0')
        else:
            mean_prob_geometric.append(prob**(1.0/(trace_length)))
            mean_prob.append(prob)
    


    # baseline length for decay
    baseline_length = sum(baseline_length) / len(baseline_length) # average length
    prob_len_ave = {}
    for k in prob_per_len.keys():
        print(k, prob_per_len[k])
        prob_len_ave[k] = sum(prob_per_len[k])/len(prob_per_len[k])
        # prob_len_ave[k] = gmean(prob_per_len[k])
    print(prob_per_len, prob_len_ave)
    print('baseline_length: ', baseline_length)

    # calculate decay factor
    last_k = 0
    decay_factor_list = [1]
    count_k = 0
    decay_factor_dic = {}
    for k in sorted(prob_len_ave):
        print(k, prob_len_ave[k])
        if count_k != 0 and k < 23: #  and k < 23
            tmp_count_degree = 1
            if k - last_k > 1:
                for i in range((k-last_k)):
                    # decay_factor_list.append(1)
                    tmp_count_degree += 1
            for nnn in range(tmp_count_degree):
                decay_factor_list.append((prob_len_ave[last_k]/prob_len_ave[k])**(1/tmp_count_degree))
                decay_factor_dic[k-nnn] = (prob_len_ave[last_k]/prob_len_ave[k])**(1/tmp_count_degree)

            print('Decay: ', (prob_len_ave[last_k]/prob_len_ave[k])**(1/tmp_count_degree))
            
        last_k = k
        count_k += 1
    print(decay_factor_dic)
    for i in sorted(decay_factor_dic.keys()):
        if i == 3:
            continue
        else:
            decay_factor_dic[i] = decay_factor_dic[i]*decay_factor_dic[i-1]

    print(decay_factor_dic)
    print('Ave Decay Factor:', sum(decay_factor_list)/len(decay_factor_list))
    print('Ave Decay Factor:', gmean(decay_factor_list))
    decay_factor = gmean(decay_factor_list)
    set_decay_factor(decay_factor)
    print('After Decay:')
    # alert_threshold = []
    for k in sorted(prob_len_ave):
        print(k, prob_len_ave[k], prob_len_ave[k]* ((decay_factor)**(k-2)), 1- prob_len_ave[k]* ((decay_factor)**(k-2)))

    deviation_score_list = []
    deviation_score_log_list = []
    for trace in trace_list:
        last_node = activity_node_dic['INITIAL']
        trace_id_list = []
        prob = 1
        trace_length = len(trace)
        # if trace_length == 13:
        #     print(trace)
        
        cur_pos_trace = 0
        largest_cur_pos_trace = -1
        # print(trace)
        trace_id_list, _ = trace_forward_wrapper(last_node[0], trace, cur_pos_trace, 1)
        # print(trace_id_list[::-1])
        if not isinstance(trace_id_list, list) or len(trace_id_list) <= 1:
            print('No transition from %s to %s  ' %(trace[largest_cur_pos_trace], trace[largest_cur_pos_trace+1]))
            # exit(1)
            continue
        # print(trace_id_list,trace)
        # exit()
        prob = calculate_probability(trace_id_list[::-1], set())
        prob2 = prob
        # prob2 = prob * decay_factor_dic[trace_length] # ((decay_factor)**(trace_length-2))

        deviation_score = 1-prob2
        deviation_score_log = 1 - math.log(prob2)
        # if deviation_score_log >=11.57:
        #     print(prob, prob2 ,deviation_score, deviation_score_log, trace)
        deviation_score_list.append(deviation_score)
        deviation_score_log_list.append(deviation_score_log)

    print("Max: ", np.max(deviation_score_list))
    print("Min: ", np.min(deviation_score_list))
    print("Mean: ", np.mean(deviation_score_list), np.std(deviation_score_list), skew(deviation_score_list), kurtosis(deviation_score_list))
    
    print("Max log:", np.max(deviation_score_log_list))
    print("Min log:", np.min(deviation_score_log_list))
    print('log: ', np.mean(deviation_score_log_list), np.std(deviation_score_log_list), skew(deviation_score_log_list), kurtosis(deviation_score_log_list))
    # mean_log = [1-math.log((1-x)) for x in deviation_score_list]
    mean_log = deviation_score_log_list
    threshold_1 = np.min(deviation_score_list)
    threshold_2 = np.max(mean_log)
    threshold_3 = np.mean(mean_log)+(3)*np.std(mean_log)
    print("Threshold 1:", threshold_1)
    print("Threshold 2:", threshold_2)
    print("Threshold 3:", threshold_3)
    set_threshold(threshold_1, threshold_2, threshold_3)
    transition_dic = copy.deepcopy(runtime_transition_dic)
    node_count_static = copy.deepcopy(node_count)


start_time = time.time()

"""
Testing
"""
test_file = './traces/trace_5fold_1'


with open(test_file, 'r') as f:
    alert_set = set()
    lines = f.readlines()

    trace_id = 0
    trace_list = []
    line_tmp = []
    trace_timestamp_id = []
    for line in lines:
        line = line.strip()
        if line.startswith('@'):
            continue
        if line.startswith('---'):
            if trace_id != 0:
                line_tmp.append('TERMINAL')
                trace_list.append(line_tmp)
            trace_timestamp_id.append(line)
            trace_id += 1
            line_tmp = []
        else:
            line_tmp.append(line.split(',')[0])
    line_tmp.append('TERMINAL')
    trace_list.append(line_tmp)
    print('Trace count %s: ' % test_file ,trace_id, len(trace_list), len(trace_timestamp_id))

    # cur_timestamp = datetime.strptime(trace_timestamp_id[0].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')

    
    trace_id = 0
    deviation_score_list = []
    transition_score_list = []
    for i in range(len(trace_list)):    # in chronological order
        trace = trace_list[i]

        initial_node = activity_node_dic['INITIAL']
        trace_id_list = []
        # prob = 1
        trace_length = len(trace) # without initial, with terminal
        # # if trace_length == 13:
        # #     print(trace)
        test_result = 0
        cur_pos_trace = 0
        largest_cur_pos_trace = -1
        print('======Trace %d' %i)

        trace_id_list, tmp_new_transition_set = trace_forward_wrapper(initial_node[0], trace, cur_pos_trace, 1)
        update_boundry = False
        print('Trace:', trace_id_list[::-1], trace_timestamp_id[i])
        if len(trace_id_list)==0:
            # print(trace_id_list,trace)
            continue
        test_result, cur_deviation_score, cur_transition_score = testing(trace_id_list[::-1], trace_timestamp_id[i], update_boundry, trace, tmp_new_transition_set)
        deviation_score_list.append(cur_deviation_score)
        for sss in cur_transition_score:
            if float(sss)>ci:
                print('#score2: ', sss)
            transition_score_list.append(sss)

        if test_result == 1:
            print('Alert: %s' % trace_timestamp_id[i])
        trace_id += 1



    cur_file_name = test_file.split('_')[-1] if not test_file.endswith('may1') else test_file.split('_')[-2]
    print('length transition_score_list: ',len(transition_score_list))
    transition_score_list = [abs(x) for x in transition_score_list]
    # plotting_cdf(transition_score_list, 'transition_5fold_%s' % cur_file_name)
    

    print('Alert set: ',len(alert_set))
    print('Alert set: ',alert_set)
    print('Trace count %s: ' % test_file ,trace_id)
    print('transition_score_list:', transition_score_list)

"""
Testing multiple files 
"""


# test_file = './traces/trace_test_may1'
# test_file = './traces/trace_train_may1'
test_file = './traces/deviation_FN_analysis_may1'
test_file_list = ['./traces/trace_train_may1', './traces/trace_test_may1', './traces/deviation_add1','./traces/deviation_add2','./traces/deviation_add3', './traces/deviation_add4', './traces/deviation_add5']
test_file_list = ['./traces/trace_5fold_training_combine', './traces/trace_5fold_testing_combine', './traces/deviation_new_add1','./traces/deviation_new_add2','./traces/deviation_new_add3', './traces/deviation_new_add4', './traces/deviation_new_add5']
test_file_list = ['./traces/trace_5fold_training_combine', './traces/trace_5fold_testing_combine']
test_file_list = ['./traces/trace_train_may1', './traces/trace_test_may1', './traces/deviation_dup1','./traces/deviation_dup2','./traces/deviation_dup3','./traces/deviation_dup4', './traces/deviation_dup5']
test_file_list = ['./traces/trace_train_may1', './traces/trace_test_may1']

deviation_score_list_list = []
transition_score_list_list = []
runtime_transition_dic_backup =  copy.deepcopy(runtime_transition_dic)
node_count_backup = copy.deepcopy(node_count)
transition_boundry_dic_backup = copy.deepcopy(transition_boundry_dic)
node_count_static_backup = copy.deepcopy(node_count_static)
transition_dic_backup = copy.deepcopy(transition_dic)
activity_node_dic_backup = copy.deepcopy(activity_node_dic)
for test_file in test_file_list:
    runtime_transition_dic = copy.deepcopy(runtime_transition_dic_backup)
    node_count = copy.deepcopy(node_count_backup)
    transition_boundry_dic = copy.deepcopy(transition_boundry_dic_backup)
    node_count_static = copy.deepcopy(node_count_static_backup)
    transition_dic = copy.deepcopy(transition_dic_backup)
    activity_node_dic = copy.deepcopy(activity_node_dic_backup)
    with open(test_file, 'r') as f:
        alert_set = set()
        lines = f.readlines()

        trace_id = 0
        trace_list = []
        line_tmp = []
        trace_timestamp_id = []
        for line in lines:
            line = line.strip()
            
            if line.startswith('---'):
                if trace_id != 0:
                    line_tmp.append('TERMINAL')
                    trace_list.append(line_tmp)
                trace_timestamp_id.append(line)
                trace_id += 1
                line_tmp = []
            else:
                line_tmp.append(line.split(',')[0])
        line_tmp.append('TERMINAL')
        trace_list.append(line_tmp)
        print('Trace count %s: ' % test_file ,trace_id, len(trace_list), len(trace_timestamp_id))

        # cur_timestamp = datetime.strptime(trace_timestamp_id[0].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')
        # cur_timestamp2 = datetime.strptime(trace_timestamp_id[0].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')
        
        trace_id = 0
        deviation_score_list = []
        transition_score_list = []
        for i in range(len(trace_list)):    # in chronological order
            trace = trace_list[i]

            initial_node = activity_node_dic['INITIAL']
            trace_id_list = []
            # prob = 1
            trace_length = len(trace) # without initial, with terminal
            # # if trace_length == 13:
            # #     print(trace)
            test_result = 0
            cur_pos_trace = 0
            largest_cur_pos_trace = -1
            # print(trace)

            trace_id_list, tmp_new_transition_set = trace_forward_wrapper(initial_node[0], trace, cur_pos_trace, 1)
            if len(tmp_new_transition_set)!=0:
                print('New Transitions: ', tmp_new_transition_set)
            # print(trace_id_list, trace_length)
            # if not isinstance(trace_id_list, list) or len(trace_id_list) <= 1:
            #     print('Alert: No transition from %s to %s  ' %(trace[largest_cur_pos_trace], trace[largest_cur_pos_trace+1]))
            #     # exit(1)
            #     continue
                # continue
            # if test_result == 0:
            update_boundry = 0

            if len(trace_id_list)==0:
                # print(trace_id_list,trace)
                continue
            test_result, cur_deviation_score, cur_transition_score = testing(trace_id_list[::-1], trace_timestamp_id[i], update_boundry, trace, tmp_new_transition_set)
            deviation_score_list.append(cur_deviation_score)
            if test_result == 1:
                print('Alert: %s' % trace_timestamp_id[i])
            trace_id += 1

            tmp_trace_id_list = trace_id_list[::-1]
            trace_transitions = set()
            for m in range(len(tmp_trace_id_list)):
                if m >= len(tmp_trace_id_list)-1:
                    continue
                trace_transitions.add((tmp_trace_id_list[m], tmp_trace_id_list[m+1]))
            
            for i in cur_transition_score:
                transition_score_list.append(i)
        
        
        if 'train' in test_file:
            cur_file_name = 'train'
        elif 'test' in test_file:
            cur_file_name = 'test'
        else:    
            cur_file_name = test_file.split('_')[-1] if not test_file.endswith('may1') else test_file.split('_')[-2]
        print('length transition_score_list: ',len(transition_score_list))
        transition_score_list = [abs(x) for x in transition_score_list]
        # plotting_cdf(transition_score_list, 'transition_synthetic_%s' % cur_file_name)
        transition_score_list_list.append(transition_score_list)
        print('Alert set: ',len(alert_set))
        print('Alert set: ',alert_set)
        print('Trace count %s: ' % test_file ,trace_id)
        # print('transition_score_list:', transition_score_list)
        # print('deviation_score_list: ', deviation_score_list)
        # plotting_cdf(deviation_score_list, 'automation_synthetic_%s' % cur_file_name)
        deviation_score_list_list.append(deviation_score_list)
# plotting_cdf_list(transition_score_list_list, 'transition_synthetic_train_test_add_threshold_5fold', test_file_list) # dup_unctrl
# plotting_cdf_list(deviation_score_list_list, 'automation_synthetic_train_test_threshold', test_file_list)
print('Threshold:', get_threshold())

exit(0)
"""
Uncontrolled dataset
"""
# Mode 1
# test_file = './traces/unctrl_jan_apr28'
test_file = './traces/unctrl_jan_may6'
new_edge_indicator = 0
print(transition_boundry_dic)
deviation_score_list = []
deviation_score_new_transition_list = []
transition_score_list = []
with open(test_file, 'r') as f:
    alert_set = set()
    lines = f.readlines()

    trace_id = 0
    trace_list = []
    line_tmp = []
    trace_timestamp_id = []
    for line in lines:
        line = line.strip()
        
        if line.startswith('---'):
            if trace_id != 0:
                line_tmp.append('TERMINAL')
                trace_list.append(line_tmp)
            trace_timestamp_id.append(line)
            trace_id += 1
            line_tmp = []
        else:
            line_tmp.append(line.split(',')[0])
    line_tmp.append('TERMINAL')
    trace_list.append(line_tmp)
    print('Trace count %s: ' % test_file ,trace_id, len(trace_list), len(trace_timestamp_id))

    cur_timestamp = datetime.strptime(trace_timestamp_id[0].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')
    cur_timestamp2 = datetime.strptime(trace_timestamp_id[0].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')
    for i in range(len(trace_list)):    # in chronological order
        trace = trace_list[i]

        initial_node = activity_node_dic['INITIAL']
        trace_id_list = []
        # prob = 1
        trace_length = len(trace) # without initial, with terminal
        # # if trace_length == 13:
        # #     print(trace)
        test_result = 0
        cur_pos_trace = 0
        largest_cur_pos_trace = -1
        # print(trace)
        trace_id_list, tmp_new_transition_set  = trace_forward_wrapper(initial_node[0], trace, cur_pos_trace, 1)
        if len(tmp_new_transition_set)!=0:
            print('New Transitions: ', tmp_new_transition_set)
        # print(trace_id_list, trace_length)
        if not isinstance(trace_id_list, list) or len(trace_id_list) <= 1:
            print('No transition from %s to %s  ' %(trace[largest_cur_pos_trace], trace[largest_cur_pos_trace+1]))
            # exit(1)
            print('Deviation score alert--', trace_timestamp_id[i])
            test_result = 2
            continue
        # if test_result == 0:
        tmp_time = datetime.strptime(trace_timestamp_id[i].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')
        if  tmp_time.timestamp() - cur_timestamp.timestamp() > 604800:
            print('NEW WEEK: ', trace_timestamp_id[i])
            cur_timestamp = tmp_time
            if tmp_time.timestamp()- cur_timestamp2.timestamp() > 2419200:
                update_boundry = 2
                cur_timestamp2 = tmp_time
            else:
                update_boundry = 1
        else:
            update_boundry = 0
        test_result, cur_deviation_score, cur_transition_score = testing(trace_id_list[::-1], trace_timestamp_id[i], update_boundry ,trace, tmp_new_transition_set)
        if new_edge_indicator == 1:
            new_edge_indicator = 0
            deviation_score_new_transition_list.append(cur_deviation_score)
        deviation_score_list.append(cur_deviation_score)
        for sss in cur_transition_score:
            transition_score_list.append(sss)

        if test_result == 1:
            print('Alert: %s\n' % trace_timestamp_id[i])
        

    # print(runtime_transition_dic)
    print('Alert set: ',len(alert_set))
    print('Alert set: ',alert_set)

print('Mode 2')
# Mode 2
# test_file = './traces/unctrl_feb_apr28'
test_file = './traces/unctrl_feb_may6'
print(transition_boundry_dic)
with open(test_file, 'r') as f:
    alert_set = set()
    lines = f.readlines()

    trace_id = 0
    trace_list = []
    line_tmp = []
    trace_timestamp_id = []
    for line in lines:
        line = line.strip()
        
        if line.startswith('---'):
            if trace_id != 0:
                line_tmp.append('TERMINAL')
                trace_list.append(line_tmp)
            trace_timestamp_id.append(line)
            trace_id += 1
            line_tmp = []
        else:
            line_tmp.append(line.split(',')[0])
    line_tmp.append('TERMINAL')
    trace_list.append(line_tmp)
    print(trace_id)

    for i in range(len(trace_list)):
        trace = trace_list[i]

        last_node = activity_node_dic['INITIAL']
        trace_id_list = []
        # prob = 1
        trace_length = len(trace)
        # # if trace_length == 13:
        # #     print(trace)
        test_result = 0
        cur_pos_trace = 0
        largest_cur_pos_trace = -1
        # print(trace)
        trace_id_list, tmp_new_transition_set = trace_forward_wrapper(last_node[0], trace, cur_pos_trace, 1)
        if len(tmp_new_transition_set)!=0:
            print('New Transitions: ', tmp_new_transition_set)
        # print(trace_id_list, trace_length)
        if not isinstance(trace_id_list, list) or len(trace_id_list) <= 1:
            print('No transition from %s to %s  ' %(trace[largest_cur_pos_trace], trace[largest_cur_pos_trace+1]))
            # exit(1)
            print('Deviation score alert--', trace_timestamp_id[i])
            test_result = 2
            continue
        
        # if test_result == 0:
        # if trace_timestamp_id[i]
        tmp_time = datetime.strptime(trace_timestamp_id[i].split('---')[1], '%Y-%m-%d %H:%M:%S.%f')
        if  tmp_time.timestamp() - cur_timestamp.timestamp() > 604800:
            print('NEW WEEK: ', trace_timestamp_id[i])
            cur_timestamp = tmp_time
            if tmp_time.timestamp()- cur_timestamp2.timestamp() > 2419200:
                update_boundry = 2
                cur_timestamp2 = tmp_time
            else:
                update_boundry = 1
        else:
            update_boundry = False

        test_result, cur_deviation_score, cur_transition_score = testing(trace_id_list[::-1], trace_timestamp_id[i], update_boundry, trace, tmp_new_transition_set)
        if new_edge_indicator == 1:
            new_edge_indicator = 0
            deviation_score_new_transition_list.append(cur_deviation_score)
        deviation_score_list.append(cur_deviation_score)
        
        for sss in cur_transition_score:
            transition_score_list.append(sss)

        
        if test_result == 1:
            print('Alert: %s\n' % trace_timestamp_id[i])

    print('Alert set: ',len(alert_set))
    print('Alert set: ',alert_set)
plotting_cdf(deviation_score_list, 'unctrl_trace')
deviation_score_list_list = [deviation_score_list, deviation_score_new_transition_list]
# plotting_cdf_list(deviation_score_list_list, 'unctrl_all_and_new_transition', ['unctrl', 'new.transition'])
print('Deviation score list:', len(deviation_score_list))


# print(transition_score_list)
transition_score_list = [abs(x) for x in transition_score_list]
plotting_cdf(transition_score_list, 'unctrl_transition')
# print('activity_node_dic:', activity_node_dic)
# print('runtime_transition_dic:', runtime_transition_dic)
# print('transition_boundry_dic:', transition_boundry_dic)
print('Threshold:', get_threshold())
print('Time:', time.time()-start_time)
