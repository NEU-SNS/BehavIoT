import os
import sys
import math
import numpy as np
import random
import copy
"""
Add/Delete/Edit events on routine testing set. 
"""
node_dic = {'amazon-plug-off': [0, 1], 'amazon-plug-on': [2], 'bulb1-color_dim': [3], 'bulb1-on_off': [4], 'dlink-camera-audio_photo_recording_watch': [5], 'echospot-on_off': [6], 'echospot-voice': [7, 8, 9], 'gosund-bulb1-on_off': [10, 11, 12], 'govee-led1-color_dim': [13], 'govee-led1-on_off': [14, 15], 'ikettle-start': [16, 17], 'magichome-strip-toggle': [18], 'meross-dooropener-close': [19], 'nest-tstat-on_off': [20], 'nest-tstat-set': [21], 'ring-camera-audio_stop_watch_motion': [22], 'ring-doorbell-audio_stop_watch_motion': [23], 'smartlife-bulb-color_dim': [24], 'smartlife-bulb-on_off': [25, 26, 27], 'switchbot-hub-on_off': [28], 't-wemo-plug-on_off': [29], 'tplink-bulb-color_dim': [30], 'tplink-bulb-on_off': [31, 32], 'tplink-plug-off': [33], 'tplink-plug-on': [34], 'wyze-cam-audio_photo_recording_watch': [35]}


# runtime_transition_dic =  {0: {2: [1.0, 1]}, 1: {1: [0.16666666666666666, 1], 9: [0.16666666666666666, 1], 24: [0.16666666666666666, 1], 33: [0.5, 3]}, 2: {1: [0.2, 1], 6: [0.2, 1], 26: [0.2, 1], 33: [0.4, 2]}, 3: {3: [0.025974025974025976, 2], 5: [0.025974025974025976, 2], 6: [0.012987012987012988, 1], 9: [0.012987012987012988, 1], 15: [0.012987012987012988, 1], 17: [0.012987012987012988, 1], 20: [0.025974025974025976, 2], 23: [0.6493506493506493, 50], 26: [0.012987012987012988, 1], 29: [0.012987012987012988, 1], 32: [0.012987012987012988, 1], 33: [0.18181818181818182, 14]}, 4: {26: [0.5, 1], 28: [0.5, 1]}, 5: {6: [0.045454545454545456, 1], 15: [0.045454545454545456, 1], 20: [0.09090909090909091, 2], 29: [0.09090909090909091, 2], 30: [0.045454545454545456, 1], 31: [0.09090909090909091, 2], 32: [0.09090909090909091, 2], 33: [0.5, 11]}, 6: {6: [0.05555555555555555, 1], 11: [0.05555555555555555, 1], 16: [0.05555555555555555, 1], 20: [0.05555555555555555, 1], 26: [0.1111111111111111, 2], 33: [0.6666666666666666, 12]}, 7: {4: [1.0, 1]}, 8: {11: [1.0, 1]}, 9: {0: [0.0196078431372549, 1], 1: [0.0392156862745098, 2], 2: [0.0784313725490196, 4], 3: [0.0196078431372549, 1], 5: [0.0196078431372549, 1], 6: [0.058823529411764705, 3], 9: [0.09803921568627451, 5], 11: [0.0196078431372549, 1], 12: [0.0196078431372549, 1], 13: [0.0392156862745098, 2], 14: [0.0784313725490196, 4], 15: [0.0392156862745098, 2], 16: [0.13725490196078433, 7], 17: [0.0196078431372549, 1], 18: [0.058823529411764705, 3], 19: [0.0196078431372549, 1], 20: [0.0196078431372549, 1], 23: [0.0196078431372549, 1], 24: [0.13725490196078433, 7], 26: [0.0196078431372549, 1], 33: [0.0392156862745098, 2]}, 10: {4: [1.0, 1]}, 11: {5: [0.043478260869565216, 1], 9: [0.30434782608695654, 7], 17: [0.043478260869565216, 1], 26: [0.043478260869565216, 1], 33: [0.5652173913043478, 13]}, 12: {10: [1.0, 1]}, 13: {9: [0.25, 1], 22: [0.25, 1], 33: [0.5, 2]}, 14: {9: [0.08333333333333333, 1], 13: [0.08333333333333333, 1], 15: [0.08333333333333333, 1], 19: [0.08333333333333333, 1], 20: [0.08333333333333333, 1], 23: [0.08333333333333333, 1], 32: [0.08333333333333333, 1], 33: [0.4166666666666667, 5]}, 15: {5: [0.05555555555555555, 1], 9: [0.05555555555555555, 1], 14: [0.05555555555555555, 1], 15: [0.2222222222222222, 4], 17: [0.05555555555555555, 1], 19: [0.1111111111111111, 2], 23: [0.05555555555555555, 1], 26: [0.05555555555555555, 1], 32: [0.05555555555555555, 1], 33: [0.2777777777777778, 5]}, 16: {20: [0.0625, 1], 26: [0.0625, 1], 27: [0.125, 2], 29: [0.3125, 5], 30: [0.25, 4], 31: [0.0625, 1], 33: [0.125, 2]}, 17: {3: [0.04, 1], 9: [0.04, 1], 15: [0.04, 1], 16: [0.04, 1], 20: [0.04, 1], 24: [0.04, 1], 32: [0.04, 1], 33: [0.72, 18]}, 18: {16: [1.0, 3]}, 19: {8: [0.08333333333333333, 1], 11: [0.9166666666666666, 11]}, 20: {3: [0.33962264150943394, 18], 21: [0.018867924528301886, 1], 23: [0.5471698113207547, 29], 26: [0.05660377358490566, 3], 33: [0.03773584905660377, 2]}, 21: {3: [0.5, 1], 32: [0.5, 1]}, 22: {26: [0.5, 1], 28: [0.5, 1]}, 23: {1: [0.009433962264150943, 1], 3: [0.33962264150943394, 36], 5: [0.02830188679245283, 3], 6: [0.018867924528301886, 2], 9: [0.018867924528301886, 2], 14: [0.009433962264150943, 1], 15: [0.018867924528301886, 2], 16: [0.009433962264150943, 1], 17: [0.018867924528301886, 2], 23: [0.2169811320754717, 23], 26: [0.03773584905660377, 4], 33: [0.27358490566037735, 29]}, 24: {3: [0.07692307692307693, 1], 24: [0.23076923076923078, 3], 26: [0.07692307692307693, 1], 33: [0.6153846153846154, 8]}, 25: {7: [1.0, 1]}, 26: {6: [0.057692307692307696, 3], 9: [0.1346153846153846, 7], 11: [0.019230769230769232, 1], 15: [0.019230769230769232, 1], 17: [0.019230769230769232, 1], 20: [0.019230769230769232, 1], 23: [0.019230769230769232, 1], 26: [0.28846153846153844, 15], 29: [0.019230769230769232, 1], 32: [0.019230769230769232, 1], 33: [0.38461538461538464, 20]}, 27: {26: [0.1111111111111111, 1], 27: [0.1111111111111111, 1], 30: [0.1111111111111111, 1], 33: [0.6666666666666666, 6]}, 28: {22: [0.5, 1], 25: [0.5, 1]}, 29: {3: [0.05, 1], 9: [0.05, 1], 11: [0.1, 2], 26: [0.1, 2], 29: [0.15, 3], 31: [0.05, 1], 33: [0.5, 10]}, 30: {5: [0.07692307692307693, 1], 17: [0.07692307692307693, 1], 26: [0.07692307692307693, 1], 27: [0.23076923076923078, 3], 29: [0.07692307692307693, 1], 32: [0.07692307692307693, 1], 33: [0.38461538461538464, 5]}, 31: {27: [0.14285714285714285, 1], 30: [0.7142857142857143, 5], 32: [0.14285714285714285, 1]}, 32: {5: [0.30434782608695654, 7], 19: [0.043478260869565216, 1], 20: [0.043478260869565216, 1], 29: [0.08695652173913043, 2], 30: [0.08695652173913043, 2], 31: [0.043478260869565216, 1], 32: [0.08695652173913043, 2], 33: [0.30434782608695654, 7]}, 34: {1: [0.005681818181818182, 1], 3: [0.09090909090909091, 16], 5: [0.03409090909090909, 6], 6: [0.03409090909090909, 6], 9: [0.13068181818181818, 23], 11: [0.03409090909090909, 6], 13: [0.005681818181818182, 1], 14: [0.03409090909090909, 6], 15: [0.028409090909090908, 5], 16: [0.017045454545454544, 3], 17: [0.09659090909090909, 17], 19: [0.03977272727272727, 7], 20: [0.23863636363636365, 42], 21: [0.005681818181818182, 1], 24: [0.005681818181818182, 1], 26: [0.08522727272727272, 15], 27: [0.011363636363636364, 2], 29: [0.028409090909090908, 5], 31: [0.011363636363636364, 2], 32: [0.0625, 11]}}
runtime_transition_dic = {0: {2: [1.0, 1]}, 1: {1: [0.125, 1], 3: [0.125, 1], 9: [0.125, 1], 28: [0.125, 1], 36: [0.5, 4]}, 2: {1: [0.2, 1], 6: [0.2, 1], 29: [0.2, 1], 36: [0.4, 2]}, 3: {1: [0.011363636363636364, 1], 3: [0.022727272727272728, 2], 5: [0.022727272727272728, 2], 6: [0.011363636363636364, 1], 9: [0.011363636363636364, 1], 18: [0.011363636363636364, 1], 20: [0.011363636363636364, 1], 23: [0.03409090909090909, 3], 27: [0.6363636363636364, 56], 29: [0.011363636363636364, 1], 32: [0.011363636363636364, 1], 35: [0.011363636363636364, 1], 36: [0.19318181818181818, 17]}, 4: {12: [0.2, 1], 27: [0.2, 1], 29: [0.2, 1], 31: [0.2, 1], 36: [0.2, 1]}, 5: {6: [0.04, 1], 9: [0.04, 1], 18: [0.04, 1], 23: [0.08, 2], 32: [0.08, 2], 33: [0.04, 1], 34: [0.12, 3], 35: [0.08, 2], 36: [0.48, 12]}, 6: {6: [0.05, 1], 9: [0.05, 1], 12: [0.05, 1], 19: [0.05, 1], 23: [0.05, 1], 29: [0.1, 2], 36: [0.65, 13]}, 7: {12: [1.0, 1]}, 8: {13: [1.0, 1]}, 9: {0: [0.014285714285714285, 1], 1: [0.02857142857142857, 2], 2: [0.05714285714285714, 4], 3: [0.014285714285714285, 1], 4: [0.014285714285714285, 1], 5: [0.014285714285714285, 1], 6: [0.04285714285714286, 3], 9: [0.07142857142857142, 5], 12: [0.014285714285714285, 1], 15: [0.08571428571428572, 6], 16: [0.014285714285714285, 1], 17: [0.05714285714285714, 4], 18: [0.04285714285714286, 3], 19: [0.11428571428571428, 8], 20: [0.02857142857142857, 2], 21: [0.04285714285714286, 3], 22: [0.014285714285714285, 1], 23: [0.014285714285714285, 1], 27: [0.02857142857142857, 2], 28: [0.12857142857142856, 9], 29: [0.014285714285714285, 1], 31: [0.014285714285714285, 1], 36: [0.12857142857142856, 9]}, 10: {4: [1.0, 2]}, 11: {32: [1.0, 1]}, 12: {5: [0.03571428571428571, 1], 9: [0.35714285714285715, 10], 20: [0.03571428571428571, 1], 29: [0.03571428571428571, 1], 36: [0.5357142857142857, 15]}, 13: {25: [1.0, 1]}, 14: {36: [1.0, 1]}, 15: {9: [0.14285714285714285, 1], 10: [0.14285714285714285, 1], 26: [0.14285714285714285, 1], 27: [0.14285714285714285, 1], 31: [0.14285714285714285, 1], 36: [0.2857142857142857, 2]}, 16: {15: [1.0, 1]}, 17: {9: [0.15384615384615385, 2], 18: [0.07692307692307693, 1], 22: [0.07692307692307693, 1], 23: [0.07692307692307693, 1], 27: [0.07692307692307693, 1], 35: [0.07692307692307693, 1], 36: [0.46153846153846156, 6]}, 18: {5: [0.05263157894736842, 1], 9: [0.05263157894736842, 1], 17: [0.05263157894736842, 1], 18: [0.21052631578947367, 4], 20: [0.10526315789473684, 2], 22: [0.10526315789473684, 2], 27: [0.05263157894736842, 1], 29: [0.05263157894736842, 1], 35: [0.05263157894736842, 1], 36: [0.2631578947368421, 5]}, 19: {23: [0.058823529411764705, 1], 29: [0.058823529411764705, 1], 30: [0.11764705882352941, 2], 32: [0.29411764705882354, 5], 33: [0.23529411764705882, 4], 34: [0.058823529411764705, 1], 36: [0.17647058823529413, 3]}, 20: {3: [0.034482758620689655, 1], 8: [0.034482758620689655, 1], 9: [0.034482758620689655, 1], 18: [0.034482758620689655, 1], 19: [0.034482758620689655, 1], 23: [0.034482758620689655, 1], 28: [0.034482758620689655, 1], 35: [0.034482758620689655, 1], 36: [0.7241379310344828, 21]}, 21: {19: [1.0, 3]}, 22: {7: [0.06666666666666667, 1], 12: [0.9333333333333333, 14]}, 23: {3: [0.3548387096774194, 22], 24: [0.03225806451612903, 2], 27: [0.5161290322580645, 32], 29: [0.06451612903225806, 4], 36: [0.03225806451612903, 2]}, 24: {1: [0.3333333333333333, 1], 3: [0.3333333333333333, 1], 35: [0.3333333333333333, 1]}, 25: {11: [1.0, 1]}, 26: {4: [0.5, 1], 10: [0.5, 1]}, 27: {1: [0.008264462809917356, 1], 3: [0.3140495867768595, 38], 5: [0.04132231404958678, 5], 6: [0.01652892561983471, 2], 9: [0.024793388429752067, 3], 17: [0.008264462809917356, 1], 18: [0.01652892561983471, 2], 19: [0.008264462809917356, 1], 20: [0.01652892561983471, 2], 27: [0.2066115702479339, 25], 29: [0.04132231404958678, 5], 32: [0.008264462809917356, 1], 36: [0.2892561983471074, 35]}, 28: {3: [0.06666666666666667, 1], 28: [0.2, 3], 29: [0.06666666666666667, 1], 36: [0.6666666666666666, 10]}, 29: {6: [0.05454545454545454, 3], 9: [0.14545454545454545, 8], 12: [0.01818181818181818, 1], 18: [0.01818181818181818, 1], 20: [0.01818181818181818, 1], 23: [0.01818181818181818, 1], 27: [0.01818181818181818, 1], 29: [0.2909090909090909, 16], 32: [0.01818181818181818, 1], 35: [0.01818181818181818, 1], 36: [0.38181818181818183, 21]}, 30: {29: [0.1111111111111111, 1], 30: [0.1111111111111111, 1], 33: [0.1111111111111111, 1], 36: [0.6666666666666666, 6]}, 31: {4: [0.3333333333333333, 1], 26: [0.3333333333333333, 1], 27: [0.3333333333333333, 1]}, 32: {3: [0.043478260869565216, 1], 9: [0.043478260869565216, 1], 12: [0.08695652173913043, 2], 29: [0.13043478260869565, 3], 32: [0.13043478260869565, 3], 34: [0.043478260869565216, 1], 36: [0.5217391304347826, 12]}, 33: {5: [0.07142857142857142, 1], 20: [0.07142857142857142, 1], 29: [0.07142857142857142, 1], 30: [0.21428571428571427, 3], 32: [0.07142857142857142, 1], 35: [0.07142857142857142, 1], 36: [0.42857142857142855, 6]}, 34: {30: [0.125, 1], 33: [0.75, 6], 35: [0.125, 1]}, 35: {5: [0.3333333333333333, 8], 22: [0.041666666666666664, 1], 23: [0.041666666666666664, 1], 32: [0.08333333333333333, 2], 33: [0.08333333333333333, 2], 34: [0.041666666666666664, 1], 35: [0.08333333333333333, 2], 36: [0.2916666666666667, 7]}, 37: {1: [0.004761904761904762, 1], 3: [0.09523809523809523, 20], 5: [0.02857142857142857, 6], 6: [0.0380952380952381, 8], 9: [0.1619047619047619, 34], 12: [0.03333333333333333, 7], 14: [0.004761904761904762, 1], 17: [0.03333333333333333, 7], 18: [0.023809523809523808, 5], 19: [0.014285714285714285, 3], 20: [0.09047619047619047, 19], 22: [0.047619047619047616, 10], 23: [0.23809523809523808, 50], 24: [0.004761904761904762, 1], 28: [0.004761904761904762, 1], 29: [0.07142857142857142, 15], 30: [0.009523809523809525, 2], 32: [0.02857142857142857, 6], 34: [0.009523809523809525, 2], 35: [0.05714285714285714, 12]}}

add_num = 1
edit_num = 2
dup_trace = 5
test_file = './traces/trace_test_may1'
test_file = './traces/trace_5fold_0'
# test_file = './traces/deviation_new_add4'
# output_file = './traces/deviation_new_add%s' % str(add_num)
# output_file = './traces/deviation_new_add5'
# output_file = './traces/deviation_edit%s' % str(edit_num)
output_file = './traces/deviation_dup%s' % str(dup_trace)

of = open(output_file, 'w')
with open(test_file, 'r') as f:
    lines = f.readlines()
    prob = 1
    last_node = ''
    mean_prob = []
    mean_prob_geometric = []
    trace_id = 0
    trace_list = []
    line_tmp = []
    for line in lines:
        line = line.strip()
        
        if line.startswith('---'):
            if trace_id != 0:
                # line_tmp.append('TERMINAL')
                trace_list.append(line_tmp)
            trace_id += 1
            line_tmp = []
            line_tmp.append(line)
        else:
            line_tmp.append(line)

    trace_list.append(line_tmp)
    print(trace_id)

    trace_fingerprint_list = []
    
    for trace in trace_list:
        tmp =  []
        for t in trace:
            if t.startswith('--'):
                continue
            tmp.append(t.split(',')[0])
        trace_fingerprint_list.append("".join(tmp))

    # # remove
    # for trace in trace_list:
    #     trace_length = len(trace)
    #     if len(trace) <= 2:
    #         continue
    #     remove_pos = random.randint(0,trace_length-2)
    #     del trace[remove_pos+1]
    #     for t in trace:
    #         of.write(t+'\n')
    
    # duplicate traces
    times = dup_trace
    for trace in trace_list:
        for i in range(times+1):
            for t in trace:
                of.write(t+'\n')
    

    # # alter
    # times = edit_num
    # for trace in trace_list:
    #     tmp =  []
    #     # tmp_trace = trace
    #     for t in trace:
    #         if t.startswith('--'):
    #             continue
    #         tmp.append(t.split(',')[0])
    #     for i in range(times):

    #         trace_length = len(trace)
    #         if len(trace) <= 2:
    #             continue
    #         while True:
    #             alter_pos = random.randint(1,trace_length-1)
    #             alter_event_num = random.randint(0,len(node_dic.keys())-1)
    #             alter_event = list(node_dic.keys())[alter_event_num]
    #             tmp[alter_pos-1] = alter_event
    #             if "".join(tmp) not in trace_fingerprint_list:
    #                 trace[alter_pos] = alter_event+', 0' 
    #                 break
    #         # if len(trace) <= 1:
    #         #     continue
    #     for t in trace:
    #         of.write(t+'\n')

    '''
    # add new transition
    times = add_num
    runtime_transition_dic_backup =  copy.deepcopy(runtime_transition_dic)
    for trace in trace_list:
        runtime_transition_dic = copy.deepcopy(runtime_transition_dic_backup)
        for i in range(times):
            # print('times:',i)
            while True:
                trace_length = len(trace)
                add_event_num = random.randint(0,len(node_dic.keys())-1)
                add_event = list(node_dic.keys())[add_event_num]
                add_pos = random.randint(0,trace_length-2)
                node_num_list = node_dic[trace[add_pos+1].split(',')[0]]
                add_event_node_num = node_dic[add_event]
                new_transition = True
                for n in node_num_list:
                    for m in add_event_node_num:
                        if m in runtime_transition_dic[n].keys():
                            new_transition = False
                            break
                if new_transition:
                    # print('New transition add:', n,m)
                    runtime_transition_dic[n][m] = [0,0]
                    break

            if len(trace) <= 1:
                continue
            count = 0
            tmp_list = []

            # print(trace, add_pos)
            for t in range(len(trace)):
                if add_pos+1 == t:
                    tmp_list = trace[t:]
                    trace[t] = add_event+', 0' 
                    break
            trace = trace[:add_pos+2] + tmp_list
            # print(trace)
        # exit(1)
        for t in trace:
            of.write(t+'\n')
    '''
    # # # add repeated
    # times = 3
    # for trace in trace_list:
        
    #     for i in range(times):
    #         while True:
    #             trace_length = len(trace)
    #             add_event_num = random.randint(0,len(node_dic.keys())-1)
    #             add_event = list(node_dic.keys())[add_event_num]
    #             add_pos = random.randint(0,trace_length-2)
    #             if runtime_transition_dic[n][m]

    #         if len(trace) <= 1:
    #             continue
    #         count = 0
    #         tmp_list = []
    #         # print(trace, add_pos)
    #         for t in range(len(trace)):
    #             if add_pos+1 == t:
    #                 tmp_list = trace[t:]
    #                 trace[t] = add_event+', 0' 
    #                 break
    #         trace = trace[:add_pos+2] + tmp_list
    #         # print(trace)
    #     # exit(1)
    #     for t in trace:
    #         of.write(t+'\n')

    # # add new transition
    # times = 9
    # runtime_transition_dic_backup =  copy.deepcopy(runtime_transition_dic)
    # for trace in trace_list:
    #     runtime_transition_dic = copy.deepcopy(runtime_transition_dic_backup)
    #     for i in range(times):
    #         # print('times:',i)
    #         while True:
    #             trace_length = len(trace)
    #             add_event_num = random.randint(0,len(node_dic.keys())-1)
    #             add_event = list(node_dic.keys())[add_event_num]
    #             add_pos = random.randint(0,trace_length-2)
    #             node_num_list = node_dic[trace[add_pos+1].split(',')[0]]
    #             add_event_node_num = node_dic[add_event]
    #             new_transition = True
    #             for n in node_num_list:
    #                 for m in add_event_node_num:
    #                     if m in runtime_transition_dic[n].keys():
    #                         new_transition = False
    #                         break
    #             if new_transition:
    #                 # print('New transition add:', n,m)
    #                 runtime_transition_dic[n][m] = [0,0]
    #                 break

    #         if len(trace) <= 1:
    #             continue
    #         count = 0
    #         tmp_list = []

    #         # print(trace, add_pos)
    #         for t in range(len(trace)):
    #             if add_pos+1 == t:
    #                 tmp_list = trace[t:]
    #                 trace[t] = add_event+', 0' 
    #                 break
    #         trace = trace[:add_pos+2] + tmp_list
    #         # print(trace)
    #     # exit(1)
    #     for t in trace:
    #         of.write(t+'\n')

    # # add not in fingerprint list
    # times = 3
    # for trace in trace_list:
        
    #     for i in range(times):
    #         tmp =  []
    #         for t in trace:
    #             if t.startswith('--'):
    #                 continue
    #             tmp.append(t.split(',')[0])
    #         tmp_trace = trace
    #         trace_length = len(trace)
    #         while True:
    #             add_event_num = random.randint(0,len(node_dic.keys())-1)
    #             add_event = list(node_dic.keys())[add_event_num]
    #             add_pos = random.randint(0,trace_length-2)
    #             if len(trace) <= 1:
    #                 continue
    #             count = 0
    #             tmp_list = []
    #             # print(trace, add_pos)
    #             for t in range(len(trace)):
    #                 if add_pos+1 == t:
    #                     tmp_list = trace[t:]
    #                     tmp_tmp = tmp[t-1:]
    #                     tmp_trace[t] = add_event+', 0' 
    #                     break
                
    #             tmp = tmp[:add_pos+1] + tmp_tmp
    #             if "".join(tmp) not in trace_fingerprint_list:
    #                 tmp_trace = tmp_trace[:add_pos+2] + tmp_list
    #                 trace = tmp_trace
    #                 break
    #         # print(trace)
    #     # exit(1)
    #     for t in trace:
    #         of.write(t+'\n')

    # # # add original
    # times = 3
    # for trace in trace_list:
        
    #     for i in range(times):
    #         trace_length = len(trace)
    #         add_event_num = random.randint(0,len(node_dic.keys())-1)
    #         add_event = list(node_dic.keys())[add_event_num]
    #         add_pos = random.randint(0,trace_length-2)
    #         if len(trace) <= 1:
    #             continue
    #         count = 0
    #         tmp_list = []
    #         # print(trace, add_pos)
    #         for t in range(len(trace)):
    #             if add_pos+1 == t:
    #                 tmp_list = trace[t:]
    #                 trace[t] = add_event+', 0' 
    #                 break
    #         trace = trace[:add_pos+2] + tmp_list
    #         # print(trace)
    #     # exit(1)
    #     for t in trace:
    #         of.write(t+'\n')

    # min_prob = 1
    # min_trans = 0
    # for k, v in runtime_transition_dic[n].items()
    #     if v[0] <= min_prob:
    #         min_prob  = v[0]
    #         min_trans = k 