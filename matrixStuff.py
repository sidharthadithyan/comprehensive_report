#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:30:50 2022

@author: thenuroh
"""
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from subprocess import call
import os
import re
import shutil

np.set_printoptions(100,sys.maxsize) 

plumeNo=1

loadData=np.load("stateAveraged_"+str(plumeNo)+".npy") #Firing rate _"+str(odourNo)+"
loadedData=np.transpose(loadData)
loadedData = np.nan_to_num(loadedData) 

plt.plot(loadedData[9,:])
plt.show

comVect = np.loadtxt(f'modules/networks/matrix_1_modules.csv',delimiter=",")

phi=np.zeros((len(comVect),4))

for i in range(len(comVect)):
    for k in range(4):
        if comVect[i]==k+1:
            phi[i,k]=1



sys.argv=['single_odor_trial.py', '59428', '1', '1', '4']

pns=[17, 44, 48, 57, 59, 65, 77, 80, 84]
#print(sys.argv)

n_n = 120
p_n = 90
l_n = 30

pPNPN = 0.0
pPNLN = 0.1
pLNPN = 0.2


pnFirnRate=loadedData[:p_n,:]
lnFirnRate=loadedData[p_n:,:]

#for i in range(90,120):


dimData=np.shape(loadedData)

ach_mat = np.zeros((n_n,n_n))

np.random.seed(64163+int(sys.argv[1])) # Random.org seed keeps the random numbers choosen in subsequent steps the save over runs
ach_mat[p_n:,:p_n] = np.random.choice([0.,1.],size=(l_n,p_n),p=(1-pPNLN,pPNLN))# :p_n stands for first p_n rows/columns p_n: stands for everything except first p_n rowns/columns
ach_mat[:p_n,:p_n] = np.random.choice([0.,1.],size=(p_n,p_n),p=(1-pPNPN,pPNPN)) #np.random.choice randowmly chooses values from the first given matrix with a probability of choice p
n_syn_ach = int(np.sum(ach_mat))


LNPN = np.zeros((p_n,l_n))

stride = int(p_n/l_n) #number of steps to be taken in later steps

spread = (round(pLNPN*p_n)//2)*2+1 # Round to closest odd integer, number of LN-PN connections


center = 0
index = np.arange(p_n) #creates a vector with entries 1 till p_n

for i in range(l_n):   
    #print((center-spread//2)%p_n)
    #print((1+center+spread//2)%p_n)
    idx = index[np.arange(center-spread//2,1+center+spread//2)%p_n] #the mod-ing helps with negetive entries when center is 0, chooses values into vectors from center - spread to center + spread

    LNPN[idx,i] = 1 #the choosen idx vales are tured into 1s

    center+=stride
    
sns.heatmap(ach_mat, cmap="RdBu")
plt.show()

fgaba_mat = np.zeros((n_n,n_n))
fgaba_mat[:p_n,p_n:] = LNPN # LN->PN LNPN is entered into fgaba matrix
fgaba_mat[p_n:,p_n:] = np.loadtxt(f'./modules/networks/matrix_{sys.argv[1]}.csv',delimiter=',') # LN->LN
np.fill_diagonal(fgaba_mat,0.)
n_syn_fgaba = int(np.sum(fgaba_mat))

#print(comVect)
pnComWegtMat=np.zeros((4,p_n))

for l in range(90):
    for k in range(90,120):
        if (ach_mat[k,l]>0):
            #print(l,k-90)
            pnComWegtMat[int((comVect[k-90])-1),l]=pnComWegtMat[int((comVect[k-90])-1),l]+1

#print(pnComWegtMat)

pnToLNCom=np.zeros((4,dimData[1]))
lnComToPN=np.zeros((4,dimData[1]))

'''print(comVect)
for i in range(4):
    print(phi[:,i])'''

for i in range(dimData[1]):
    pnToLNCom[:,i]=np.matmul(pnComWegtMat,pnFirnRate[:,i])
    
    for k in range(4):
        lnComToPN[k,i]=np.matmul(phi[:,k],lnFirnRate[:,i])

print(np.max(pnToLNCom))
print(np.max(lnComToPN))
plt.plot(pnToLNCom[1,:])
plt.plot(lnComToPN[1,:])
plt.show()
            
sns.heatmap(fgaba_mat, cmap="RdBu")

sgaba_mat = np.zeros((n_n,n_n))
sgaba_mat[:p_n,p_n:] = LNPN
np.fill_diagonal(sgaba_mat,0.)
n_syn_sgaba = int(np.sum(sgaba_mat))
plt.show()

plt.figure(figsize=(7,6))
sns.heatmap(sgaba_mat, cmap="RdBu")
plt.xlabel("Presynaptic Neuron Number")
plt.ylabel("Postsynaptic Neuron Number")
plt.title("Network Connectivity")
plt.show()

'''for l in range(9):
    for k in range(80,120):
        if (sgaba_mat[pns[l],k]==1):
            flag=0
            for h in range(90,120):
                if (sgaba_mat[h,k]==1):
                    flag=flag+1
                    print(flag)
                    
for l in range(90):
    gig=0
    for k in range(90,120):
        if (ach_mat[k,l]>0):
            flag=0
            for h in range(90):
                if (ach_mat[k,h]==1):
                    for r in range(9):
                        if (pns[r]==h):
                            flag=flag+1
            print(pns[l],k-90,flag)
            #print(pns[l],k,flag)
   # print(pns[l],gig)
                    
                    
                    '''