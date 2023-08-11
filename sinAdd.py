#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:45:46 2023

@author: thenuroh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

x = np.arange(0, 100, 0.1)

sins = np.zeros((5, len(x)))


sins[0, :] = np.sin(x)
sins[1, :] = np.sin(5*x)
sins[2, :] = np.sin(x/10)
sins[3, :] = np.sin(x/20)
sins[4, :] = np.sin(x/50)


def fit_func(x, *args):
    x=np.arange(len(x))
    #sins1N=sins[0, :]
   # sins1N=np.append(np.zeros(int(args[0])), sins1N)
    #sinsN=sins1N[0:len(sins[0, :])]
    print(args[2])
    
    return args[1]*sins[0, x-int(args[0])]+args[2]*sins[1, x]+args[3]*sins[2, x]+args[4]*sins[3, x]+args[5]*sins[4, x]

h = fit_func(x, 100,5,2,30,6,9)

p,cov = opt.curve_fit(fit_func,x,h,p0=[10,1,0.5,0.5,0,0], maxfev=1000000000)
  
print(p)

plt.plot(h)
plt.plot(fit_func(x, *p))
plt.show()
