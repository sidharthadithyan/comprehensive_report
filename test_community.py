#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:34:03 2022

@author: thenuroh
"""

import numpy as np

mat1 = np.loadtxt(f'/home/thenuroh/Documents/linear_model/modules/networks/matrix_1_modules.csv',delimiter=",")

print(mat1)
phi=np.zeros((len(mat1),4))

for i in range(len(mat1)):
    for k in range(4):
        if mat1[i]==k+1:
            phi[i,k]=1

print(phi)