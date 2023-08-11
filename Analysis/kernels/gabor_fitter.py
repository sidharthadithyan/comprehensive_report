#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:59:52 2022

@author: thenuroh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


for g in [1,2]:
    odorNo=1
    plumeNo=g    
    #The subset of PNs perturbed
    pns=[17, 44, 48, 57, 59, 65, 77, 80, 84]
    
    noPNs=len(pns)
    noBin=2000
    binSize=1
    #Matrix to store parameters
    paraSetDir=np.zeros((noPNs,6))
    paraSetIndir=np.zeros((noPNs,noBin))#*binSize
    kernelStore=np.zeros((noPNs,noBin*binSize))
    
    
    #Load firing rate and plume input
    plumeDataRaw=np.load("plume/data_plume_"+str(plumeNo)+".npy") #Plume input
    
    
    plumeData=np.zeros(len(plumeDataRaw))
    shiftDur=100
    for i in range(len(plumeData)-shiftDur):
        plumeData[i]=plumeDataRaw[shiftDur+i]
    
    #loadGuessPulse=np.load("guessPulse/stateGuessAveraged_"+str(plumeNo)+".npy")
    #loadedData=np.transpose(loadGuessPulse)
    loadData=np.load("firing_rate/stateAveraged_"+str(odorNo)+"_"+str(plumeNo)+".npy") #Firing rate _"+str(odourNo)+"
    loadedData=np.transpose(loadData)
    
    loadedData = np.nan_to_num(loadedData) 
    
    for i in range(noPNs):
        #number of the PN being analysed 
        pnNumber=pns[i] #[17, 44, 48, 57, 59, 65, 77, 80, 84]
        
        
        
        #Choosing the particular PN from the imported data
        sample_trace=loadedData[pnNumber,:]
        sample_trace=sample_trace#[4250:7000]
        sample_odor=plumeData#[4250:7000]
        print(np.shape(sample_trace))
        #Visualizing the data
        xAxis=np.arange(len(sample_odor)) #for ploting and visualization
        plt.figure(figsize=(15,3))
        plt.plot(xAxis,sample_odor, color='black')
        plt.plot(xAxis,sample_trace, color='red')
        plt.show()
        
        
        
        
        # Define a gabor function
        def gabor(x,*args):
            """
            Gabor function
            """
            gaussian = np.exp(-0.5*(x-args[0])**2/args[1]**2)
            return args[4]*np.cos(2*np.pi*args[2]*(x-args[0])+args[3])*gaussian+args[5]
        
        
        
        
        
        '''
        "Direct Method"
        # define the function to fit
        def fit_func(x,*args):
            gabor_filter = gabor(np.linspace(0,1,noBin),*args)
            return np.convolve(x,gabor_filter,mode='full')[:len(x)]
        
        # fit the data
        p,cov = opt.curve_fit(fit_func,sample_odor,sample_trace,p0=[0,1,0.5,0.5,0,0], maxfev=1000000000)
             
        # plot the fit
        plt.plot(sample_trace[::10],label='data')
        plt.plot(fit_func(sample_odor,*p)[::10],label='fit')
        plt.legend()
        plt.show()
        
        # plot the gabor filter
        plt.plot(gabor(np.linspace(0,1,noBin),*p)/np.max(gabor(np.linspace(0,1,noBin),*p)),label='fit')
        plt.legend()
        plt.show()
        
        paraSetDir[i,:]=p'''
        
    
        
        "Indirect Method"
        
        # define the function to fit
        def fit_func_2(x,*args):
            gabor_filter = np.repeat(np.array(args),binSize)
    
            return np.convolve(x,gabor_filter,mode='full')[:len(x)]
        
        p0=np.zeros(noBin)
        #for j in range(noBin):
    
            #p0[i]=np.mean(loadGuessPulse[400+(j*50):400+((j+1)*50),pnNumber])
            #print(np.sum(p0))
    
        #plt.plot(loadGuessPulse[400:400+2000,pnNumber],label='guess' )
        #plt.show()
        # fit the data
        p,cov = opt.curve_fit(fit_func_2,sample_odor,sample_trace,p0, maxfev=1000000000)
        
        residuals = sample_trace- fit_func_2(sample_odor, *p)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((sample_trace-np.mean(sample_trace))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        
        
        #print(cov)
        # plot the fit
        f = plt.figure()
        f.set_figwidth(15)
        f.set_figheight(2)
        plt.plot(sample_trace[::50],label='data')
        plt.plot(fit_func_2(sample_odor,*p)[::50],label='fit')
        plt.legend()
        plt.show()
        #f.savefig('PN_'+str(pns[i])+'.png', dpi=300)
        
        
        # plot the gabor filter
        f = plt.figure()
        plt.plot(np.repeat(p,binSize)/np.max(p),label='fit')
        
        binSizeFltr=50
        print(len(p))
        noBinFltr=int(len(p)/binSizeFltr)
        kernelFltr=np.zeros(noBinFltr)
        for m in range(noBinFltr):
            kernelFltr[m]=np.mean(p[m*binSizeFltr:m*binSizeFltr+binSizeFltr])
        print(len(kernelFltr))
        plt.plot(np.repeat(kernelFltr, binSizeFltr)/np.max(kernelFltr),label='filtr')
            
            
        
        
        # fit gabor filter to the best fit and plot
        def fit_gabor(x,*args):
            return gabor(x,*args)
        
        p_,cov_ = opt.curve_fit(fit_gabor,np.linspace(0,1,noBin*binSize),np.repeat(p,binSize),p0=[0,1,0.5,0.5,0,0], maxfev=1000000000)
        #plt.plot(gabor(np.linspace(0,1,noBin*binSize),*p_)/np.max(gabor(np.linspace(0,1,noBin*binSize),*p_)),label='gabor fit')
        plt.legend()
        plt.show()
        #f.savefig('Kernel_'+str(pns[i])+'.png', dpi=300)
        
        paraSetIndir[i,:]=p
        kernelStore[i,:]=np.repeat(kernelFltr, binSizeFltr)
        
    np.save(f'parameters_'+str(plumeNo)+'_direct',paraSetDir)
    
    np.save(f'parameters_'+str(plumeNo)+'_indirect',paraSetIndir)
    
    np.save(f'kernel_'+str(plumeNo)+'_indirect',kernelStore)
    np.savetxt('kernel_'+str(plumeNo)+'_indirect.csv',kernelStore,delimiter=',')
