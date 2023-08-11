import numpy as np
import sys
import matplotlib.pyplot as plt

#Inputs to generate plume
for u in range(5):
    for v in range(5):
        plumeNo=u+1
        odorNo=v+1
        shiftPlume = False
        sys.argv=['4','4',odorNo,plumeNo]
         
        #parameters for resolution and block size
        blocktime = 12000 # in ms
        sim_res = 1 # in ms
        min_block = 50 # in ms
        
        #Control over the fine-ness of plume layers
        np.random.seed(int(sys.argv[3]))
        switch_prob = 0.4
        if switch_prob == 0.0:
            sw_state = [1]
        else:
            sw_state = [0]
        
        #Generating the intitial string
        flag_print=1
        rand_string=np.random.choice([0,1],p=[1-switch_prob,switch_prob],size=int(blocktime/min_block)-1)
        
        #makes a random sequence of ones and zeros the shape of blocktime/min blocktime
        for i in rand_string: 
            if flag_print==1:
                flag_print=0
            if i==1:
                sw_state.append(1-sw_state[-1])
            else:
                sw_state.append(sw_state[-1])
        
        #Generating the plume from the string
        sw_state=np.pad(sw_state, (10, 10), 'constant', constant_values=(0, 0)) #add buffer of zeros to both ends of the string
        ts = np.repeat(sw_state,int(min_block/sim_res)) #Expands the sequenc to get the turbulant fluid stream
        
        if shiftPlume == True:
                
            tsN=ts
            pushCurr=300
            tsN = np.append(np.zeros(pushCurr), tsN)
            ts=tsN[0:len(ts)]
            
            #Ploting and visualization
            xAxis=np.arange(len(ts))
            plt.figure(figsize=(15,3))
            plt.plot(xAxis,ts, color='black')
            plt.savefig('figures/Plume_shifted_'+str(plumeNo)+'.svg') 
            plt.show()
            
            #saving the plume
            np.save(f'data_plume_shifted_'+str(sys.argv[3]),ts)
            np.savetxt('data_plume_shifted_'+str(sys.argv[3])+'.csv',ts,delimiter=',')
        
        else:
            
            #Ploting and visualization
            xAxis=np.arange(len(ts))
            plt.figure(figsize=(15,3))
            plt.plot(xAxis,ts, color='black')
            #plt.savefig('figures/Plume_'+str(plumeNo)+'.svg') 
            plt.show()
            
            #saving the plume
            np.save(f'data_plume_'+str(sys.argv[3]),ts)
            np.savetxt('data_plume_'+str(sys.argv[3])+'.csv',ts,delimiter=',')
