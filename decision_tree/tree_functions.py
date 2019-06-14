# -*- coding: utf-8 -*-
"""
Created on Thu Jun  13 22:32:21 2019

@author: 29132
"""

import numpy as np

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    m=len(dataSet)
    labelcounts={}
    for i in range(m):
        label=dataSet[i][-1]
        labelcounts[label]=labelcounts.get(label,0)+1
    shannonEnt = 0.0
    for key in labelcounts.keys():
        pro=labelcounts[key]/m
        shannonEnt-=(pro*np.log2(pro))
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    ret_dataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reduced_FeatVec=featVec[:axis]
            reduced_FeatVec.extend(featVec[axis+1:])
            ret_dataSet.append(reduced_FeatVec)
    return ret_dataSet

def chooseBestFeatureToSplit(dataSet):
    n_features=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInforgain=0.0
    bestFeatures=-1
    for i in range(n_features):
        featlist=[example[i] for example in dataSet]
        uniqvalues=set(featlist)
        new_entrop=0.0
        for value in uniqvalues:
            subdataSet=splitDataSet(dataSet,i,value)
            pro=len(subdataSet)/len(dataSet)
            new_entrop+=pro*calcShannonEnt(subdataSet)
        Inforgain=baseEntropy-new_entrop
        if Inforgain>bestInforgain:
            bestInforgain=Inforgain
            bestFeatures=i
    return bestFeatures


def majorityCnt(classList):
    classcounts={}
    for label in classList:
        classcounts[label]=classcounts.get(label,0)
    sorted_counts=sorted(classcounts.items(),key=lambda x:x[1],reverse=True)
    return sorted_counts[0][0]

#==ID3 algorithm===================
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeatures=chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel=labels[bestFeatures]
    my_tree={bestFeatureLabel:{}}
    del labels[bestFeatures]
    featureValue=[example[bestFeatures] for example in dataSet]
    uniqValues=set(featureValue)
    for value in uniqValues:
        sublabels=labels[:]
        subdataSet=splitDataSet(dataSet, bestFeatures, value)
        my_tree[bestFeatureLabel][value]=createTree(subdataSet,sublabels)
    return my_tree


import matplotlib.pyplot as plt
decisionNode=dict(boxstyle='sawtooth',fc='y')
leafNode=dict(boxstyle='round4',fc='y')
arrow_args = dict(arrowstyle="<-")

"""
plt.annotate(s,xy,xycoords,xytext,textcoords,va,ha,bbox,arrowprops)
s:str,the text of annotation
xy:the point(x,y) to annotate
xytest:The position (x,y) to place the text at. If None, defaults to xy.
xycoords:Defaults to the value of xycoords, 
            i.e. use the same coordinate system for annotation point and text position.
textcoords:The coordinate system that xytext is given in
arrowprops:
1. If arrowprops contains the key 'arrowstyle' the above keys are forbidden. 
The allowed values of 'arrowstyle' are:'->'
2. If arrowprops does not contain the key 'arrowstyle' the allowed keys are:
    width,headwidth,headlength,shrink,?
va:verticalalignmen
ha:horizontalalignment    
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt, textcoords='axes fraction',\
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure()
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
    

def getNumLeafs(my_tree):
    numLeafs=0
    firstLeaf=list(my_tree.keys())[0]
    secondDict=my_tree[firstLeaf]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(my_tree):
    maxDepth=0
    firstr=list(my_tree.keys())[0]
    secondDict=my_tree[firstr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': \
                                                 {0: 'no', 1: 'yes'}}}},
    {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]
my_tree=retrieveTree(0)
numLeafs=getNumLeafs(my_tree)
depth=getTreeDepth(my_tree)
    
"""
plt.text(x,y,s):
x, y : scalars
The position to place the text. By default, this is in data coordinates. 
The coordinate system can be changed using the transform parameter
s:str, the text
"""

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

createPlot(my_tree)
    
        
            
            
    


