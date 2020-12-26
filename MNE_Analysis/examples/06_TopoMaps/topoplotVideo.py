import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

##Software is removing last epoch of data
##Solution create events manually

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

#Read eeg file
file = Path('./../data/juan_S3_T2_epoc_pyprep.edf')
raw = mne.io.read_raw_edf(file)

#Rename Channel
mne.rename_channels(raw.info, renameChannels)
#Set montage (3d electrode location)
raw = raw.set_montage('standard_1020')

#Create events every 20 seconds
events_array = mne.make_fixed_length_events(raw, start=1, stop=None, duration=2)
# events_array = np.vstack((events_array , [67500+5000,0,1]))

#Get 20 seconds Epochs from data
epochs = mne.Epochs(raw, events_array, tmin=-0.95, tmax=0.95, preload=True)
fig, axes = plt.subplots(1,3)
fig.patch.set_facecolor('grey')

for i in range(len(epochs)):
    #Select only first four epochs
    singleEpoch = epochs[i]
    singleEpoch.load_data()

    #Plot sensor location
    # singleEpoch.plot_sensors(kind='topomap', ch_type='all', show_names=True)

    #Frequency spatial distributions
    # bands_list = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 45, 'Gamma')]
    # bands_list = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta') ]
    bands_list = [ (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta') ]
    epochFig = singleEpoch.plot_psd_topomap(bands=bands_list ,ch_type='eeg', normalize=True, show=False,\
                                             vlim=(0.0,0.5), cmap='RdBu_r')

    #Transform figure to image
    img = np.frombuffer(epochFig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(epochFig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
    plt.close(epochFig)

    # display image with opencv or any operation you like
    cv2.imshow("plot", img)


    k = cv2.waitKey(0)


cv2.destroyAllWindows()