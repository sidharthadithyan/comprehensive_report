import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from tqdm import tqdm

# Change to reflect your file location!
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\spsha\\Desktop\\ffmpeg-4.4-full_build\\bin\\ffmpeg.exe'

#metadata = dict(title='Movie', artist='codinglikemad')
writer = PillowWriter(fps=15) #, metadata=metadata)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

noPlumes=3
noOdors=5
noComun=3

stichPlume=False

#plumeStichedLen=np.zeros(noPlumes)
lnToPNDict={}

if stichPlume:
    for i in range(noPlumes):
        plume=np.load("data_plume_"+str(i+1)+".npy")
        plumeStiched=[]
        plumeStichedLen=0
        for k in range(len(plume)):
            if plume[k]>0:
                plumeStiched=np.append(plumeStiched,k)
        np.save(f'plumeStiched_'+str(i),plumeStiched)
        plumeStichedLen=len(plumeStiched)
        lnToPNHolder=np.zeros((noOdors,noComun,plumeStichedLen))
        for j in range(noOdors):
            lnComToPNRaw=np.load("lnComToPNNormalized_"+str(j+1)+"_"+str(i+1)+".npy")
            print(lnComToPNRaw.shape)
            lnComToPNStich=np.array([[],[],[]])
    
            for l in range(plumeStichedLen):
                lnComToPNStich=np.append(lnComToPNStich,lnComToPNRaw[:,int(plumeStiched[l])].reshape(3,1), axis=1)
    
    
            lnToPNHolder[j,:,:]=lnComToPNStich
            print(lnToPNHolder.shape)
        lnToPNDict.update({"Plume_"+str(i): lnToPNHolder})
else:
    for i in range(noPlumes):
        
        arrays = [np.load("lnComToPNNormalized_"+str(j+1)+"_"+str(i+1)+".npy") for j in range(noOdors)]
        lnToPNHolder=np.stack(arrays, axis=0)
        print(lnToPNHolder.shape)
        lnToPNDict.update({"Plume_"+str(i): lnToPNHolder})
    

with writer.saving(fig, "exp32d.gif", 100):
    for tval in tqdm(range(1000,1100)):
        colorcount=0
        for i in range(noPlumes):
            lnToPNHolderL= lnToPNDict["Plume_"+str(i)]
            for j in range(noOdors):
      
                lnComToPN=lnToPNHolderL[j,:,:]
                #print(lnComToPN.shape)
                plt.xlim(0, 1000)
                plt.ylim(0, 1000)
                ax.set_zlim(0,1000)
                
                
                color=["#BA0000", "#CB2323", "#DD4545", "#BA9800","#CBAA23","#DDBC45", "#03800E","#299C31","#4EB755",  "#034580", "#28699B", "#4E8DB7",  "#790380", "#95289B", "#B24EB7"]

            
                ax.plot(lnComToPN[0,tval-5:tval], lnComToPN[1,tval-5:tval], lnComToPN[2,tval-5:tval],c=color[colorcount])
                ax.scatter(lnComToPN[0,tval], lnComToPN[1,tval], lnComToPN[2,tval],c=color[colorcount])
                
                colorcount=colorcount+1
                
        writer.grab_frame()
        plt.cla()