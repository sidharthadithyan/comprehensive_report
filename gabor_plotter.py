#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:39:46 2022

@author: thenuroh
"""
import numpy as np
import matplotlib.pyplot as plt
    
#The subset of PNs perturbed
pns=[17, 44, 48, 57, 59, 65, 77, 80, 84]

noPNs=len(pns)

#Matrix to store parameters
paraSetPlume1=np.zeros((noPNs,6))
paraSetPlume2=np.zeros((noPNs,6))

paraSetPlume1=np.load("parameters_1.npy")
paraSetPlume2=np.load("parameters_2.npy")

def gabor(x,*args):
    """
    Gabor function
    """
    gaussian = np.exp(-0.5*(x-args[0])**2/args[1]**2)
    return args[4]*np.cos(2*np.pi*args[2]*(x-args[0])+args[3])*gaussian+args[5]

for i in range(noPNs):
    #number of the PN being analysed 
    pnNumber=pns[i] #[17, 44, 48, 57, 59, 65, 77, 80, 84]
    
    p1=paraSetPlume1[i,:]
    p2=paraSetPlume2[i,:]
    
    # plot the gabor filter
    plt.plot(gabor(np.linspace(0,1,500),*p1)/np.max(gabor(np.linspace(0,1,500),*p1)),label='Plume 1')
    plt.plot(gabor(np.linspace(0,1,500),*p2)/np.max(gabor(np.linspace(0,1,500),*p2)),label='Plume 2')
    plt.legend()
    plt.show()