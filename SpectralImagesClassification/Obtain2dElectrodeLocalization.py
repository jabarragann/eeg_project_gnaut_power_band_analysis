from pathlib import Path
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math as m
from SpectralImagesClassification.SpectralImagesUtils import azim_proj

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]
#Load a sample file
file = Path('./data/UJing_S02_T01_Baseline_raw.edf')
raw = mne.io.read_raw_edf(file)
mne.rename_channels(raw.info, renameChannels)
raw = raw.set_montage('standard_1020') #Set montage (3d electrode location)

locations = mne.channels.make_standard_montage('standard_1020')

listOfLocations = []

for elec in locations.dig:
    listOfLocations.append([elec['ident'],elec['kind'],elec['coord_frame'],elec['r']])

listOfLocations = np.array(listOfLocations)
listOfLocations = listOfLocations[3:]

ch_names = np.array(locations.ch_names)
finalLocationList3D = pd.DataFrame(columns=['ch_name','x','y','z'])
finalLocationList2D = pd.DataFrame(columns=['ch_name','x','y'])

for count, ch in enumerate(EEG_channels):
    idx = np.where(ch_names == renameChannels(ch))

    try:
        idx = idx[0][0]
    except:
        print("{:} was not found in the list of channels",ch)

    finalLocationList3D.loc[count, 'ch_name'] = ch
    finalLocationList3D.loc[count, ['x', 'y', 'z']] = listOfLocations[idx,3]

    finalLocationList2D.loc[count, 'ch_name'] = ch
    finalLocationList2D.loc[count, ['x', 'y']] = azim_proj(listOfLocations[idx, 3].astype(np.float))

#Save 2D locations
finalLocationList2D.to_csv("channel_2d_location.csv",sep=',',index=False)

#Plot channels
#3D
# coor = finalLocationList3D[['x','y','z']].values.astype(np.float)
# xs,ys,zs = coor[:,0],coor[:,1],coor[:,2]
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
# ax.scatter3D(xs,ys,zs, color='green')
# plt.show()
#2D
# xs, ys = finalLocationList2D['x'].values, finalLocationList2D['y'].values
# fig2, ax2 = plt.subplots(1,1)
# ax2.scatter(xs,ys)
# plt.show()

