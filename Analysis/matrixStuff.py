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
import scipy.stats as st

np.set_printoptions(100,sys.maxsize) 

for u in range(5):
    for v in range(5):
        plumeNo=v+1
        odorNo=u+1
        
        plumeDataRaw=np.load("data_plume_"+str(plumeNo)+".npy")
        loadData=np.load("stateAveraged_"+str(odorNo)+"_"+str(plumeNo)+".npy") #Firing rate _"+str(odourNo)+"
        loadedData=np.transpose(loadData)
        loadedData = np.nan_to_num(loadedData) 
        
        #plt.plot(loadedData[9,:])
        #plt.show
        
        
        comVect = np.loadtxt(f'modules/networks/matrix_4_modules.csv',delimiter=",")
        
        phi=np.zeros((len(comVect),3))
        
        for i in range(len(comVect)):
            for k in range(3):
                if comVect[i]==k+1:
                    phi[i,k]=1
        
        normPhi=np.zeros(3)
        
        for i in range(3):
            normPhi[i]=np.sum(phi[:,i])
        
        print("normPhi")
        print(normPhi)
        
        sys.argv=['single_odor_trial.py', '4', odorNo, plumeNo, '4']
        
        if odorNo==1:
            pns=[17, 44, 48, 57, 59, 65, 77, 80, 84]
        elif odorNo==2:
            pns=[9, 10, 23, 29, 30, 51, 63, 71, 82]
        #print(sys.argv)
        
        n_n = 120
        p_n = 90
        l_n = 30
        
        pPNPN = 0.0
        pPNLN = 0.1
        pLNPN = 0.2
        
        
        pnFirnRate=loadedData[:p_n,:]
        lnFirnRate=loadedData[p_n:,:]
        
        lnToPnWgt=np.zeros((3,9))
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
            
        '''sns.heatmap(ach_mat, cmap="RdBu")
        plt.show()'''
        
        fgaba_mat = np.zeros((n_n,n_n))
        fgaba_mat[:p_n,p_n:] = LNPN # LN->PN LNPN is entered into fgaba matrix
        fgaba_mat[p_n:,p_n:] = np.loadtxt(f'./modules/networks/matrix_{sys.argv[1]}.csv',delimiter=',') # LN->LN
        np.fill_diagonal(fgaba_mat,0.)
        n_syn_fgaba = int(np.sum(fgaba_mat))
        
        #print(comVect)
        pnComWegtMat=np.zeros((3,p_n))
        
        for l in range(90):
            for k in range(90,120):
                if (ach_mat[k,l]>0):
                    #print(l,k-90)
                    pnComWegtMat[int((comVect[k-90])-1),l]=pnComWegtMat[int((comVect[k-90])-1),l]+1
        
        for l in range(9):
            for k in range(90,120):
                if (fgaba_mat[l,k]>0):
                    lnToPnWgt[int((comVect[k-90])-1),l]=lnToPnWgt[int((comVect[k-90])-1),l]+1
                
        print(lnToPnWgt)
        #print(pnComWegtMat)
        
        pnToLNCom=np.zeros((3,dimData[1]))
        lnComToPN=np.zeros((3,dimData[1]))
        
        
        pnToLNComSmooth=np.zeros((3,dimData[1]+399))
        lnComToPNSmooth=np.zeros((3,dimData[1]+399))
        lnComToPNNormSmooth=np.zeros((3,dimData[1]+399))
        
        
        '''print(comVect)
        for i in range(4):
            print(phi[:,i])'''
            
        for i in range(dimData[1]):
            pnToLNCom[:,i]=np.matmul(pnComWegtMat,pnFirnRate[:,i])
            
            for k in range(3):
                lnComToPN[k,i]=np.matmul(phi[:,k],lnFirnRate[:,i])
        
        x=np.arange(-200,200)
        args=[0,50]
        gker= np.exp(-0.5*(x-args[0])**2/args[1]**2)
        
        for i in range(3):
            pnToLNComSmooth[i,:]=np.convolve(gker, pnToLNCom[i,:])
            lnComToPNSmooth[i,:]=np.convolve(gker, lnComToPN[i,:])
            
            print(np.max(pnToLNComSmooth))
            print(np.max(lnComToPNSmooth))
            
            r=st.pearsonr(pnToLNComSmooth[i,:], lnComToPNSmooth[i,:]) 
            print(r)
            '''plt.plot(pnToLNComSmooth[i,:])
            plt.plot(lnComToPNSmooth[i,:],label='fit'+'_R^2 ='+str(r),color="red")
            plt.show()'''
                      
        '''plt.plot(lnComToPNSmooth[0,:],label='fit'+'_R^2 ='+str(r),color="red")
        plt.plot(lnComToPNSmooth[1,:],label='fit'+'_R^2 ='+str(r),color="blue")
        plt.plot(lnComToPNSmooth[2,:],label='fit'+'_R^2 ='+str(r),color="green")
        plt.show()'''
           
        
        pnInh=np.zeros((9,dimData[1]))
        
        for i in range(9):
            pnInh[i,:]=np.matmul(np.transpose(lnComToPN),lnToPnWgt[:,i])
            
        '''for i in range(9):
            plt.plot(pnInh[i,:],label="pn"+str(i))
            plt.show()'''
        
        res=np.zeros((30,dimData[1]))
        lnPred=np.zeros((30,dimData[1]))
        
        print(np.shape(phi[0]))
        
        lnComToPNNormalized=np.zeros((3,dimData[1]))
        
        for i in range(dimData[1]):
            lnComToPNNormalized[0,i]=lnComToPN[0,i]/normPhi[0]
            lnComToPNNormalized[1,i]=lnComToPN[1,i]/normPhi[1]
            lnComToPNNormalized[2,i]=lnComToPN[2,i]/normPhi[2]
                   
            lnPred[:,i]=(lnComToPN[0,i]/normPhi[0]*phi[:,0]+lnComToPN[1,i]/normPhi[1]*phi[:,1]+lnComToPN[2,i]/normPhi[2]*phi[:,2])
            res[:,i]=lnFirnRate[:,i]-lnPred[:,i]
        
        
        
        for i in range(3):
            lnComToPNNormSmooth[i,:]=np.convolve(gker, lnComToPNNormalized[i,:])
            
            #print(np.max(pnToLNComSmooth))
            #print(np.max(lnComToPNSmooth))
            
            #r=st.pearsonr(pnToLNComSmooth[i,:], lnComToPNSmooth[i,:]) 
            #print(r)
            #plt.plot(pnToLNComSmooth[i,:])
            #plt.plot(lnComToPNSmooth[i,:],label='fit'+'_R^2 ='+str(r),color="red")
            #plt.show()
        
        
        
        print(lnFirnRate[:,2])
        '''plt.plot(lnFirnRate[2,:])
        plt.plot(lnPred[2,:])
        plt.show()
        
        
        plt.plot(lnComToPNNormSmooth[0,:],label='fit'+'_R^2 ='+str(r),color="#003f5c")
        plt.plot(lnComToPNNormSmooth[1,:],label='fit'+'_R^2 ='+str(r),color="#ffa600")
        plt.plot(lnComToPNNormSmooth[2,:],label='fit'+'_R^2 ='+str(r),color="#bc5090")
        plt.show()'''
        
        color=["#003f5c","#ffa600","#bc5090","#ff6361"]
        fig, axs = plt.subplots(3,1)
        fig.suptitle('LN to PN Community firing rates odor '+str(odorNo)+" plume "+str(plumeNo))
        for j in range (0,3):
            #axs[j].axis(xmin=0.5,xmax=13.5)
            #axs[j].axis(ymin=0.5,ymax=13.5)
            axs[j].plot(lnComToPNNormSmooth[j,500+200:8000+200],color=color[j],label="x"+str(j+1))
            axs[j].plot(plumeDataRaw[500:8000]*1000,color="#58508d",label="x"+str(j+1))
            axs[j].grid()
        
        np.save(f'lnComToPNNormalized_'+str(odorNo)+'_'+str(plumeNo),lnComToPNNormSmooth)
        plt.savefig('Figure/'+str(odorNo)+'_'+str(plumeNo))
        
        '''
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
        plt.show()'''
        
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
