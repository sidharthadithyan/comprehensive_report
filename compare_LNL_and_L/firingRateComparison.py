#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 22:54:52 2023

@author: thenuroh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Define a gabor function
def gabor(x,*args):
    """
    Gabor function
    """
    gaussian = np.exp(-0.5*(x-args[0])**2/args[1]**2)
    return args[4]*np.cos(2*np.pi*args[2]*(x-args[0])+args[3])*gaussian+args[5]

# test gabor function
x = np.linspace(0,1,100)
y = gabor(x,0.5,0.2,1,np.pi/2,1,1)
plt.plot(x,y)
plt.show()




def generate_pn_activity(length,history_length,noise_level,intermittency,interval=50):
    """
    A function to generate a single trace of PN activity

    Variables:
        length: int
            the length of the trace in ms
        history_length: int
            the length of gabor history in ms
        noise_level : float
            The noise level of the PN activity
        intermittency : float
            The intermittency of the odor trace

    Returns:
        activity : array
            The PN activity
    """
    trace_length = int(length/interval)
    history_length = int(history_length/interval)
    # Generate a random odor trace
    odor_trace = np.zeros(trace_length)
    for i in range(1,trace_length):
        if np.random.rand() < intermittency:
            odor_trace[i] = 1-odor_trace[i-1]
        else:
            odor_trace[i] = odor_trace[i-1]
    # Add noise
    activity = odor_trace + np.random.randn(trace_length)*noise_level
    # create random gabor pattern
    gabor_trace = gabor(np.linspace(0,1,history_length),0.2,0.1,2,np.pi/2,1,0.2)
    
    c2=0
    for i in range(len(gabor_trace)):
        c2=c2+(gabor_trace[i]*gabor_trace[i])
    
    const=np.sqrt(c2)
    
    
    #normalize
    gabor_trace_norm = gabor_trace/const
    
    # convolve with odor trace
    activity = np.convolve(activity,gabor_trace_norm,mode='full')[:trace_length]
    # add noise
    activity = activity
    # keep only positive values
    activity[activity<0] = 0
    
    gain=const
    c=(gain*2)
    print(c)
    t=np.arange(-2,2,0.001)
    plt.plot(t,gain*t+c)
    
    plt.show()
    
    for i in range(len(activity)):
        if activity[i]>0:
            activity[i]=(activity[i]*gain)+c
    # normalize
    #activity = activity/np.max(activity)
    
    return activity,gabor_trace_norm,gabor_trace,odor_trace,trace_length

sample_trace,sample_gabor_norm,sample_gabor,sample_odor,trace_length = generate_pn_activity(10000,500,0.1,0.005,1)
# plot a sample trace
#plt.plot(sample_trace)

plt.plot(np.convolve(sample_odor,sample_gabor,mode='full')[:trace_length])
plt.savefig("Figures/firingRate.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.plot(sample_odor)
plt.savefig("Figures/plume.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.plot(sample_gabor)
plt.savefig("Figures/kernel.pdf", format="pdf", bbox_inches="tight")
plt.show()
