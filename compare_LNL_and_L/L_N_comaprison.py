#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:36:07 2023

@author: thenuroh
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

plumeNo=2

#Load firing rate and plume input
plumeDataRaw=np.load("data_plume_"+str(plumeNo)+".npy") #Plume input

t=np.arange(0,2,0.001)

def pOne(t):
    toh=0.08
    n=5
    a=0.8
    k = (((t/toh)**n)*np.exp(-t/toh))-(a/2)*((t/2*toh)**n)*np.exp(-t/2*toh)
    return k/np.max(k)
    
def pTwo(t):
    toh=0.08
    n=5
    a=0.8
    k=np.diff((((t/toh)**n)*np.exp(-t/toh))-(a/2)*((t/2*toh)**n)*np.exp(-t/2*toh))
    return k/np.max(k)
    
pOne=pOne(t)
pTwo=np.array(pTwo(t))

pTwoL=np.zeros(np.shape(pOne))
pTwoL[0:len(pTwo)]=pTwo

plt.plot(t,pOne)
plt.plot(t,pTwoL)
plt.show()


inputPlume=np.zeros(5000)
inputPlume[500:4500]=0.0001

plt.plot(inputPlume)
plt.show()



inputPlume=np.zeros(5000)
inputPlume[500:4500]=0.0001

plt.plot(inputPlume)
plt.show()


angles=(np.arange(0,360,45))*np.pi/180

print(angles)

filterStore=np.zeros((len(angles),len(pOne)))
count=0
firingRates=np.zeros((len(angles),14999))
for theta in angles:
    y=np.sin(theta)
    x=np.cos(theta)
    filterTrail=(x*pOne)+(y*pTwoL)
    filterStore[count,:]=filterTrail
    firingRates[0,:]=np.convolve(filterTrail,plumeDataRaw)
    print(np.shape(firingRates[0,:]))
    
    plt.plot(firingRates[0,:])
    count=count+1

plt.show()

noPNs=len(angles)

for i in range(noPNs):
    #number of the PN being analysed 
    #pnNumber=pns[i] #[17, 44, 48, 57, 59, 65, 77, 80, 84]
    
    
    
    #Choosing the particular PN from the imported data
    sample_trace=firingRates[0,0:13000]
    sample_odor=plumeDataRaw
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
    

    
    "Indirect Method"
    
    noBin=2000
    binSize=1
    
    
    
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
    p,cov = opt.curve_fit(fit_func_2,sample_odor,sample_trace,p0, maxfev=1000)
    
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
        
        
    

