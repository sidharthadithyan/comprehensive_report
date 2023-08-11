#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:35:51 2023

@author: thenuroh
"""
import numpy as np

import matplotlib.pyplot as plt




for k in range(30):
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    for i in range(2):
        for j in range(2):
                    
            odorNo=i+1
            plumeNo=j+1
            
            lnComToPNRaw=np.load("lnComToPNNormalized_"+str(odorNo)+"_"+str(plumeNo)+".npy")
            
            plume=np.load("data_plume_"+str(plumeNo)+".npy")
            
            
            lnComToPNStich=[]
            
            count=0
            
            for g in range(len(plume)):
                if plume[g]>0:
                    lnComToPNStich=np.append(lnComToPNStich,lnComToPNRaw[:,g])
                    count=count+1
            
            lnComToPN=lnComToPNStich.reshape(3,count)
            
            color=["red", "orange", "blue", "green"]
            
            ax.scatter(lnComToPN[0,:][::10], lnComToPN[1,:][::10], lnComToPN[2,:][::10])
    
    
            ax.view_init(0, k)
            fig