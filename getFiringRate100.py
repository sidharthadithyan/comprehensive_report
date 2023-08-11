import matplotlib.pyplot as plt
import numpy as np

##Parameters of size and resolution of data
sim_res = 0.01 # in ms
min_block = 50 # in ms
sim_time = 13000

odourNo=1

#Information about the experiment
n_n=120
n_trial=100
n_trial_x=10
n_trial_y=10

#Loading the raw data
loadData=np.zeros((n_trial,sim_time,n_n))
for k in range(0,n_trial_x):
    for i in range(0,n_trial_y):
        loadData[(10*k+(i)),:,:]=np.load("/home/thenuroh/Documents/Data/odour_"+str(odourNo)+"_data/data_59428_1_"+str(odourNo)+"_"+str(i+1)+"_"+str(k+1)+".npy")
        
#temporary matrix to store data where each AP is represented with a 1 and rest of values are zero
temp1Matx=np.zeros((np.shape(loadData)[1],np.shape(loadData)[2])) #(13000,120)

#converting data into boolean and adding all trials into one string. Each 1 corresponds to an action potential in some trial 
for k in range(1,n_trial):
    for i in range(np.shape(temp1Matx)[1]): #120
        for j in range(np.shape(temp1Matx)[0]): #13000
            if j==0:
                temp1Matx[j,i]=0
            elif loadData[k,j-1,i]<0 and loadData[k,j,i]>=0: #threshold condition for counting AP as a spike
                temp1Matx[j,i]=temp1Matx[j,i]+1 #adding spikes over all 10 odors

#temporary matrix to store firing rates
temp2Matx=np.zeros(np.shape(temp1Matx))

#Size of sliding window for rate calculation 
windowSize = 50

#Calculating the rate
for i in range(np.shape(temp1Matx)[1]): #120
    for j in range(np.shape(temp1Matx)[0]): #1300010
        if np.shape(temp1Matx)[0]-j>windowSize:
            temp2Matx[j,i]=np.average(temp1Matx[j:j+windowSize,i]) #average of total bin size
        else:
            temp2Matx[j,i]=np.average(temp1Matx[j:-1,i]) #average over remaining points when time points are less than total bin width
 
temp2Matx=np.true_divide(temp2Matx,n_trial)

#Saving firing rates
np.savetxt('stateAveraged_'+str(odourNo)+'.csv',temp2Matx,delimiter=',')
np.save(f'stateAveraged_'+str(odourNo),temp2Matx)

#print which all neurons has non zero firing rate (Subset perturbed)
indexHolder=[]
for i in range(90):    
    if np.average(temp2Matx[0:3000,i])>0:
        indexHolder.append(i)
print(indexHolder)

#plot firing rate of any neuron
t = np.arange(0,sim_time,sim_res)
plt.plot(t[0:13000],temp2Matx[:,44])