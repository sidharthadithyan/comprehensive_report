#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:45:46 2023

@author: thenuroh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

x = np.arange(0, 1000, 0.1)
xN = np.arange(0, 1000, 0.01)
newSin=np.zeros((len(xN)))

sins = np.zeros((5, len(x)))

def gaussian(x,*args):
    return np.exp(-0.5*(x-args[0])**2/args[1]**2)

print(gaussian(11, 10, 2))

sins[0, :] = np.sin(x)
sins[1, :] = np.sin(5*x)
sins[2, :] = np.sin(x/10)
sins[3, :] = np.sin(x/20)
sins[4, :] = np.sin(x/50)


def newSin(x):
    n=1
    sigma=n/2
    xVal=0
    wights=0
    for j in range(int(x)-n,int(x)+n):
        #print(int(xN[i]))
        xVal=xVal+((sins[0,int(j)])*gaussian(int(j), int(x), sigma))
        wights=wights+gaussian(int(j), int(x), sigma)
    #print(wights)    
    return xVal/wights
    

plt.plot(sins[0,:])
#plt.plot(xN,newSin(xN))
plt.show()

def fit_func(x, *args):
    x=np.arange(len(x))
    y=x-args[0]
    valZero=np.zeros((len(y)))
    for i in range(len(y)):
        valZero[i]=newSin(y[i])
    return args[1]*valZero[x]+args[2]*sins[1, x]+args[3]*sins[2, x]+args[4]*sins[3, x]+args[5]*sins[4, x]

h = fit_func(x,12,5,2,30,6,9)

p,cov = opt.curve_fit(fit_func,x,h,p0=[10,1,0.5,0.5,0,0], maxfev=1000000000)
  
print(p)

plt.plot(h)
plt.plot(fit_func(x, *p))
plt.show()
