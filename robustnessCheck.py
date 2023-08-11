#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:34:38 2022

@author: thenuroh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

#The subset of PNs perturbed
pns=[17, 44, 48, 57, 59, 65, 77, 80, 84]


noBin=40
binSize=50

noPNs=len(pns)
plumeNo=2
plumeNoP=1
#Matrix to store parameters
paraSet=np.zeros((noPNs,6))
shiftDur=100
print(paraSet)
#Load firing rate and plume input
plumeDataRaw=np.load("data_plume_"+str(plumeNo)+".npy") #Plume input
plumeData=np.zeros(np.shape(plumeDataRaw))
print(np.shape(plumeData))

loadData=np.load("stateAveraged_"+str(plumeNo)+".npy") #Firing rate
loadedData=np.transpose(loadData)
print(np.shape(loadedData))
#loadedData = np.ones(np.shape(loadedData))*100

loadedData = np.nan_to_num(loadedData) 

paraSet=np.load("kernel_"+str(plumeNo)+"_indirect.npy")
print(np.shape(paraSet))

for i in range(len(plumeData)-shiftDur):
    plumeData[i]=plumeDataRaw[shiftDur+i]

for i in range(noPNs):
    #number of the PN being analysed 
    pnNumber=pns[i] #[17, 44, 48, 57, 59, 65, 77, 80, 84]
    
    p=paraSet[i,:]
    
    #Choosing the particular PN from the imported data
    sample_trace=loadedData[pnNumber,:]
    sample_trace=sample_trace#[4250:7000]
    sample_odor=plumeData#[4250:7000]
    
    # Define a gabor function
    def gabor(x,*args):
        """
        Gabor function
        """
        gaussian = np.exp(-0.5*(x-args[0])**2/args[1]**2)
        return args[4]*np.cos(2*np.pi*args[2]*(x-args[0])+args[3])*gaussian+args[5]
    
    
    
    
    
    
    '''"Direct Method"
    # define the function to fit
    def fit_func(x,*args):
        gabor_filter = gabor(np.linspace(0,1,500),*args)
        return np.convolve(x,gabor_filter,mode='full')[:len(x)]
    
    
    # plot the fit
    plt.plot(sample_trace[::10],label='data')
    plt.plot(fit_func(sample_odor,*p)[::10],label='fit')
    plt.legend()
    plt.show()
    
    # plot the gabor filter
    plt.plot(gabor(np.linspace(0,1,500),*p)/np.max(gabor(np.linspace(0,1,500),*p)),label='fit')
    plt.legend()
    plt.show()'''
    
    "Indirect Method"
    # define the function to fit
    def fit_func_2(x,*args):
        gabor_filter = args#np.repeat(np.array(args),binSize)
        print(len(gabor_filter))
        return np.convolve(x,gabor_filter,mode='full')[:len(x)]
    
    residuals = sample_trace- fit_func_2(sample_odor, *p)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sample_trace-np.mean(sample_trace))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    plt.plot(sample_trace[::10],label='data')
    plt.plot(fit_func_2(sample_odor,*p)[::10],label='fit R^2 ='+str("%.2f" % r_squared))
    plt.legend()
    plt.show()
    
    fitted=fit_func_2(sample_odor,*p)
    # fit the data
    df1 = pd.DataFrame(sample_trace)
    df2 = pd.DataFrame(fitted)
    
    
    df11=df1.rolling(window=100).mean()
    df22=df2.rolling(window=100).mean()
    
    sample_t=df11.to_numpy()
    fted=df22.to_numpy()
    
    
    fted[np.isnan(fted)] = 0
    sample_t[np.isnan(sample_t)] = 0
    #print(fted)
    residuals = sample_t-fted
    ss_res = np.sum(residuals**2)
    #print("line")
    #print(ss_res)
    ss_tot = np.sum((sample_t-np.mean(sample_t))**2)
    #print(ss_tot)
    r_squared = 1 - (ss_res / ss_tot)
    
    #print(type(p))
    
    buffer=np.zeros(500)
    #print(type(buffer))
    p2=np.append(buffer,p)
    kerData=pd.DataFrame(p2)
    kerDataMean=kerData.rolling(window=150).mean()
    kerSmooth=kerDataMean.to_numpy()
    
    x=np.arange(-200,200)
    args=[0,50]
    gker= np.exp(-0.5*(x-args[0])**2/args[1]**2)
    
    kerSmooth=np.convolve(gker, p)
    '''p2=np.zeros(np.shape(p))
    windowSize = 50
    for j in range(np.shape(p)[0]): #1300010
        if np.shape(p)[0]-j>windowSize:
            p2[j]=np.average(p[j:j+windowSize]) #average of total bin size
        else:
            p2[j]=np.average(p[j:-1])'''
            
    r=st.pearsonr(sample_t[:,0], fted[:,0]) 
    print(r)        
    plt.plot(kerSmooth, label='fit'+'_R^2 =',color='#DC267F')
    plt.show()
    
    fig, ax = plt.subplots(3,2,figsize=(16,4), gridspec_kw={'width_ratios': [1, 8]})
    
    
    ax[0,0].legend()
    ax[0,0].set_title('Kernels_PN_'+str(pns[i]))
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,0].set_ylabel('Frequency (Hz)')
    ax[0,0].plot(kerSmooth, label='fit'+'_R^2 ='+str(r),color='#DC267F')
    
    
    ax[0,1].legend()
    ax[0,1].set_title('Fit'+str(r))
    ax[0,1].set_xlabel('Time (ms)')
    ax[0,1].set_ylabel('Frequency (Hz)')
    ax[0,1].plot(sample_t[::10],label='data'+str(plumeNo),color='#648FFF')
    ax[0,1].plot(fted[::10],label='fit'+'_R^2 ='+str(r_squared),color='#DC267F')
    plt.savefig('figures/line_plot'+str(i)+'.svg') 
    
    plt.show()
    