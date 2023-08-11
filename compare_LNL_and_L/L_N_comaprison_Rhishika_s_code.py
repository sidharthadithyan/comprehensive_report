#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 22:37:44 2023

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
    
    #normalize
    gabor_trace = gabor_trace/np.max(gabor_trace)
    
    # convolve with odor trace
    activity = np.convolve(activity,gabor_trace,mode='full')[:trace_length]
    # add noise
    activity = activity
    # keep only positive values
    activity[activity<0] = 0
    
    gain=20
    c=gain*2
    t=np.arange(-2,2,0.001)
    plt.plot(t,gain*t+c)
    
    plt.show()
    
    for i in range(len(activity)):
        if activity[i]>0:
            activity[i]=(activity[i]*gain)+c
    # normalize
    #activity = activity/np.max(activity)
    
    return activity,gabor_trace,odor_trace

sample_trace,sample_gabor,sample_odor = generate_pn_activity(10000,500,0.1,0.005,1)
# plot a sample trace
plt.plot(sample_trace)
plt.plot(sample_odor)
plt.show()
plt.plot(sample_gabor)
plt.show()



def fit_func_2(x,*args):
    gabor_filter = np.array(args)
    return np.convolve(x,gabor_filter,mode='full')[:len(x)]

# fit the data
p,cov = opt.curve_fit(fit_func_2,sample_odor,sample_trace,p0=np.zeros(500))

# plot the fit
plt.plot(sample_trace[::10],'.',label='data')
plt.plot(fit_func_2(sample_odor,*p)[::10],label='fit')
plt.legend()
plt.show()

plt.plot(sample_odor)
plt.show()
# plot the gabor filter
plt.plot(sample_gabor,'.',label='data')
plt.plot(p/np.max(p),label='fit')

# fit gabor filter to the best fit
def fit_gabor(x,*args):
    return gabor(x,*args)

#p_,cov_ = opt.curve_fit(fit_gabor,np.linspace(0,1,500),p,p0=[0,1,0.5,0.5,0,0])
#plt.plot(gabor(np.linspace(0,1,500),*p_),label='gabor fit')
plt.legend()
plt.show()

