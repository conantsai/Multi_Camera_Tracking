import datetime
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import tensorflow as tf

from siamese_model import *
from siamese_dataset import *

np.set_printoptions(suppress=True)

def createNode():
    nodes = list()
    nodes_datetime = list()

    with open('data/cam1/tracker.txt', 'r') as readline:
        cam1_data = readline.readlines()

    for index, content in enumerate(cam1_data):
        tracklet, time = content.split(":")
        begin_time, end_time = time.strip().strip().split(",")
        begin_time = datetime.datetime.strptime(time.strip().strip().split(",")[0], "%M'%S'%f")
        end_time = datetime.datetime.strptime(time.strip().strip().split(",")[1], "%M'%S'%f")
        content = [int(tracklet), begin_time, end_time]
        cam1_data[index] = content

    with open('data/cam2/tracker.txt', 'r') as readline:
        cam2_data = readline.readlines()

    for index, content in enumerate(cam2_data):
        tracklet, time = content.split(":")
        begin_time, end_time = time.strip().strip().split(",")
        begin_time = datetime.datetime.strptime(time.strip().strip().split(",")[0], "%M'%S'%f")
        end_time = datetime.datetime.strptime(time.strip().strip().split(",")[1], "%M'%S'%f")
        content = [int(tracklet), begin_time, end_time]
        cam2_data[index] = content

    for i, cam1 in enumerate(cam1_data):
        for j, cam2 in enumerate(cam2_data):
            cam1Arr = cam1[1]
            cam1Lea = cam1[2]
            cam2Arr = cam2[1]
            cam2Lea = cam2[2]
            if (cam1Arr < cam2Arr) and (cam1Lea < cam2Lea):
                if cam1Lea + datetime.timedelta(minutes=1) > cam2Arr:
                    nodes.append([cam1[0], cam2[0]])  
                    nodes_datetime.append([[cam1[1], cam1[2]], [cam2[1], cam2[2]]])
         
    cam1Groups = createGroups(cam1_data, 6)
    cam2Groups = createGroups(cam2_data, 6)

    return nodes, nodes_datetime, cam1Groups, cam2Groups

def createGroups(tracklet, timeInterval):
    group = [ [None] * len(tracklet) for i in range(len(tracklet)) ]

    for i in range(0, len(tracklet)):
        Lea = tracklet[i][2]
        group[i][i] = 0
        for j in range(i+1, len(tracklet)):
            Arr = tracklet[j][1]
            if Arr + datetime.timedelta(seconds=timeInterval) < Lea:   
                group[i][j] = 1
                group[j][i] = 1
            else: 
                group[i][j] = 0
                group[j][i] = 0

    return np.array(group) 

def constructGraph(nodes, group1, group2):
    count = len(nodes)
    graph = np.zeros(shape=(count, count))
    for i in range(0, count):
        for j in range(i, count):
            if (nodes[i][0] != nodes[j][0]) and (nodes[i][1] != nodes[j][1]) and (nodes[i][0] != nodes[j][1]) and (nodes[i][1] != nodes[j][0]) \
                                                and (isGroup(group1, nodes[i][0], nodes[j][0]) or isGroup(group2, nodes[i][1], nodes[j][1])):
                graph[i][j] = 1
                graph[j][i] = 1
    return graph

def isGroup(group, tracklet1, tracklet2):
    if tracklet1 > len(group)-1: tracklet1 -= len(group)
    if tracklet2 > len(group)-1: tracklet2 -= len(group)
    if group[tracklet1][tracklet2] == 1:
        return True
    else:
        return False

def constructConflict(nodes):
    count = len(nodes)
    conflict = np.zeros(shape=(count,count))
    for i in range(0, count):
        for j in range(i+1, count):
            if nodes[i][0] == nodes[j][0] or nodes[i][1] == nodes[j][1] or nodes[i][0] == nodes[j][1] or nodes[i][1] == nodes[j][0]:
                conflict[i][j] = 1
                conflict[j][i] = 1
    return conflict

def getEcost(graph, nodes, nodes_datetime):
    np.set_printoptions(threshold=np.nan)
    nodeCount = len(nodes)
    ecost = np.zeros((nodeCount, nodeCount))
    interval_info = np.zeros((nodeCount))


    for i in range(nodeCount):
        for j in range(nodeCount):
            if graph[i][j] == 1:
                lea = nodes_datetime[j][0][1]
                arr = nodes_datetime[j][1][0]
                interval = (arr - lea)
                interval = interval.total_seconds()
                interval_info[j] = interval
        break
    interval_mean = np.mean(interval_info)
    interval_std = np.std(interval_info, ddof = 0)

    for i in range(nodeCount):
        for j in range(nodeCount):
            if graph[i][j] == 1:
                lea = nodes_datetime[j][0][1]
                arr = nodes_datetime[j][1][0]
                interval = (arr - lea)
                interval = interval.total_seconds()
                ans = norm.pdf(x = interval, loc=interval_mean, scale=interval_std)
                ecost[i][j] = ans
            else:
                ecost[i][j] == 0
    return ecost

def getImage(filepath, colorTrans=False):
    image_value = io.imread(filepath)
    if colorTrans == True:
        if source_flag == True: 
            Slab = color.rgb2lab(image_value)
            SLABmean = Slab.mean(0).mean(0)
            SLABstd = Slab.std(0).std(0)
            source_flag = False
        else:
            Tlab = color.rgb2lab(image_value)
            TLABmean = Tlab.mean(0).mean(0)
            TLABstd = Tlab.std(0).std(0)
            height, width, channels = image_value.shape
            for x in range(0, height):
                for y in range(0, width):
                    Tlab[x][y][0] = (SLABstd[0]/TLABstd[0])*(Tlab[x][y][0] - TLABmean[0]) + SLABmean[0]
                    Tlab[x][y][1] = (SLABstd[1]/TLABstd[1])*(Tlab[x][y][1] - TLABmean[1]) + SLABmean[1]
                    Tlab[x][y][2] = (SLABstd[2]/TLABstd[2])*(Tlab[x][y][2] - TLABmean[2]) + SLABmean[2]
            Trgb = color.lab2rgb(Tlab)
            image_value = (255.0 / Trgb.max() * (Trgb - Trgb.min())).astype(np.uint8)
    image_value = [image_value/255]
    image_value = np.array(image_value)

    return image_value
    
def getVcost(nodes):
    tf.reset_default_graph()

    colorTrans = False
    vcost = np.zeros((len(nodes)))
    vnode = list()
    img_placeholder = tf.placeholder(tf.float32, [None, 128, 64, 3], name='img')
    net = siamese(img_placeholder, is_training=False, reuse=False)
    saver = tf.train.Saver()
    # net = siamese2(img_placeholder)
    # saver = tf.train.import_meta_graph("model/model.ckpt.meta")

    for index, node in enumerate(nodes):
        filepath1 = "data/cam1/cam1_person" + str(node[0]) + ".png"
        filepath2 = "data/cam2/cam2_person" + str(node[1]) + ".png"

        cam1 = getImage(filepath1)
        cam2 = getImage(filepath2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "model/model.ckpt")
            cam1_feat = sess.run(net, feed_dict={img_placeholder:cam1})
            cam2_feat = sess.run(net, feed_dict={img_placeholder:cam2})

        vcost_dist = cdist(cam1_feat, cam2_feat, 'euclidean')
        vcost[index] = vcost_dist
        vnode.append([index, node[0], node[1]])
        if index == 0:
            print(node, vcost_dist)

    vcost_list = np.argsort(vcost, axis=0)
    vcost = np.sort(vcost, axis=0)
    vnode_list = list()
    for i in vcost_list:
        vnode_list.append(vnode[i])

    return vcost, np.array(vnode_list)

def resultToDict(nodeUse):
    result = {i[0][0]: i[0][1] for i in nodeUse}
    return result

def calculateAccuracy(real, result):
    allPair = len(real.keys())
    resultPair = len(result.keys())
    correct = 0
    notMatch = 0
    wrongMatch = 0

    for key in real:
        if key in result and real[key] == result[key]:
            correct+=1
        elif key in result and real[key] != result[key]:
            wrongMatch+=1
        else :
            notMatch+=1
    if resultPair > allPair:
        wrongMatch += (resultPair-allPair)
    
    mme = wrongMatch
    tp = correct
    if resultPair > allPair:
        wrongMatch += (resultPair-allPair)

    if tp == 0:
        MCTA = None
    else:
        MCTA = 1-(mme/tp)

    print('right mapping: ' + str(correct))
    print('wrong mapping: ' + str(notMatch + wrongMatch))
     
    print('mme: ' + str(wrongMatch))
    print('tp: ' + str(correct))
    print('MCTA: ' + str(MCTA))

    return correct/allPair

if __name__ == '__main__':
    nodes, nodes_datetime, group1, group2 = createNode()
    graph = constructGraph(nodes, group1, group2)
    conflict = constructConflict(nodes)
    nodeCount = len(nodes)

    ecost = getEcost(graph, nodes, nodes_datetime)
    vcost, vlist = getVcost(nodes)

    # print(nodes)
    # print(vlist)
    # print(graph)
    # print(conflict)
    # print(vcost, vlist)

    # vcost = np.array([0.28168089, 0.31964477, 0.32662189, 0.35603418, 0.36034267, 0.36462012,
    #                   0.38390326, 0.38678372, 0.39294957, 0.39694841, 0.40760776, 0.40787301,
    #                   0.41503042, 0.42973736, 0.43110009, 0.4389519,  0.43998377, 0.44437953,
    #                   0.45231508, 0.45783457, 0.46180174, 0.47365255, 0.48974594, 0.49202142,
    #                   0.49764192, 0.54727578]) 

    # vlist = np.array([[8, 2, 8], [18, 4, 10], [13, 3, 10], [2, 0, 8], [22, 5, 8],  [7, 2, 7],
    #                   [6, 2, 6], [17, 4, 8],  [10, 3, 7],  [21, 5, 7], [5, 1, 8],  [11, 3, 8],
    #                   [1, 0, 7], [15, 4, 6],  [4, 1, 7],  [3, 1, 6], [14, 3, 11], [19, 4, 11],
    #                   [20, 5, 6], [12, 3, 9],  [16, 4, 7],  [23, 5, 9], [25, 5, 11], [24, 5, 10],
    #                   [0, 0, 6], [9, 3, 6]])

    maxLabel = 0
    maxCost = 1000
    
    nodeUse = list()
    edgeUse = list()

    ## matching algorithm
    for index1, content1 in enumerate(vlist):
        label = [1] * nodeCount
        nodeTemp = list()
        edgeTemp = list()
        label[index1] = 1
        nodeTemp.append([nodes[content1[0]], vcost[index1]])

        ## whether conflict with other nodes
        for index2, content2 in enumerate(vlist):
            if conflict[content1[0]][content2[0]] == 1:
                label[index2] = 0

        ## whether node is legal
        for index2, content2 in enumerate(vlist):
            if label[index2] == 1 and index2 != index1:
                if graph[content1[0]][content2[0]] == 1:
                    print(nodes[content2[0]])
                    nodeTemp.append([nodes[content2[0]], vcost[index2]])
                    edgeTemp.append([nodes[content1[0]], nodes[content2[0]], ecost[content1[0]][content2[0]]])
                    ## Is there any other nodes that conflicts with new add node
                    for index3, content3 in enumerate(vlist):
                        if conflict[content2[0]][content3[0]] == 1:
                            label[index3] = 0

    
        labels = len(nodeTemp)
        if labels > maxLabel:
            maxLabel = labels
            nodeCost = 0
            edgeCost = 0

            for i in nodeTemp:
                nodeCost += i[1]
            for i in edgeTemp:
                edgeCost += i[2]
            cost = nodeCost  

            if cost <= maxCost:
                maxCost = cost
                nodeUse = nodeTemp
                edgeUse = edgeTemp
                break

        elif labels == maxLabel:
            nodeCost = 0
            edgeCost = 0

            for i in nodeTemp:
                nodeCost += i[1]
            for i in edgeTemp:
                edgeCost += i[2]
            cost = nodeCost  

            if cost <= maxCost:
                maxCost = cost
                nodeUse = nodeTemp
                edgeUse = edgeTemp

    real = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
    
    ## with Unary cost & Pairwise cost result
    result = dict()
    result = resultToDict(nodeUse)
    print("-------------------------with Unary cost & Pairwise cost result-------------------------")
    print(result)
    accuracy = calculateAccuracy(real, result)

    ## with Unary cost result
    sn_used = list()
    sn_nodeUse = list()
    for index, content in enumerate(vlist):
        if content[1] not in sn_used and content[2] not in sn_used:
            sn_used.append(content[1])
            sn_used.append(content[2])
            sn_nodeUse.append([nodes[content[0]], vcost[index]])

    sn_result = dict()
    sn_result = resultToDict(sn_nodeUse)
    print("-------------------------with Unary cost result-------------------------")
    print(sn_result)
    sn_accuracy = calculateAccuracy(real, sn_result)



