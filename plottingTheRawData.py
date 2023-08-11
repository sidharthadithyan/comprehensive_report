import matplotlib.pyplot as plt
import numpy as np

#Parameters of size and resolution of data
sim_time = 13000 #in ms
sim_res = 0.01 # in ms

t = np.arange(0,sim_time,sim_res)

#Loading data
dataNo=9
loadData=np.zeros((13000,120))
loadData=np.load("/home/thenuroh/Documents/Data/data_59428_1_1_"+str(dataNo)+".npy")

#Plotting
fig,ax = plt.subplots(1,1,figsize=(12,6))
for i in range(90):
    plt.plot(i*0.02+t[::100]/1000,-i*10+loadData[:,i],linewidth=1,color=plt.cm.inferno(i/90))
plt.hlines(-900,0.5,1.0,color='k',linewidth=3)
plt.vlines(0.5,-900,-800,color='k',linewidth=3)
plt.text(0.5,-960,"0.5 s",fontdict={"fontsize":18})
plt.text(0.2,-910,"100 mV",fontdict={"fontsize":18},rotation=90)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,15)
plt.box(False)
plt.tight_layout()

#Saving the Plot
#plt.savefig(f"IntemittentPNOutput.svg")

