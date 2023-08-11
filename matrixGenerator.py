#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:21:35 2022

@author: thenuroh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n = 30
p = 1
A = np.random.choice([0,1],p=[1-p,p],size=(n,n))
A = np.int32((A+A.T)>0)
np.fill_diagonal(A,0)
for i in range(30):
    for j in range(30):
        if i<10 and j<10:
            A[i,j]=0
        if i>=10 and i<20 and j>=10 and j<20:
            A[i,j]=0
        if i>19 and j>19:
            A[i,j]=0

sns.heatmap(A, cmap="RdBu")        

np.save(f'matrix_',A)
np.savetxt(f'matrix_.csv',A,delimiter=',')
