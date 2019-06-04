# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:22:10 2019

@author: 29132
"""

import numpy as np
import pandas as pd

data=pd.read_csv('watermelon_3a.csv')

def calc_entropy(dataSet):
    m=len(dataSet)
    labelcounts={}
    for i in range(m):
        label=dataSet[i][-1]
        labelcounts[label]=labelcounts.get(label,0)+1
    entropy=0.0
    for counts in labelcounts.values():
        prob=counts/m
        entropy-=prob*np.log2(prob)
    return entropy

dataset=list(data.values)
entropy=calc_entropy(dataset)
