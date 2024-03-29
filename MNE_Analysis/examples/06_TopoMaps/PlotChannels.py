import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
raw = mne.io.read_raw_edf(file, preload=True)

#Pick significant Channels
# ch_names = ['AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8',
#             'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'CP5', 'CP6']
ch_names = ['F7','F4','F8','FC5','FC1','FC2','FC6','T7','C4','CP5']

raw = raw.pick_channels(ch_names)
#Rename Channel
mne.rename_channels(raw.info, renameChannels)
#Set montage (3d electrode location)
raw = raw.set_montage('standard_1020')

#Create events every 20 seconds
events_array = mne.make_fixed_length_events(raw, start=10, stop=None, duration=20)
events_array = np.vstack((events_array , [67500+5000,0,1]))

#Get 20 seconds Epochs from data
epochs = mne.Epochs(raw, events_array, tmin=-9.5, tmax=9.5)

#Select only first four epochs
# epochs = epochs[:4]
epochs.load_data()

#Plot sensor location
epochs.plot_sensors(kind='topomap', ch_type='all', show_names=True)

# #Frequency spatial distributions
# bands_list = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 45, 'Gamma')]
# bands_list = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta') ]
# epochs.plot_psd_topomap(bands=bands_list ,ch_type='eeg', normalize=True)
#
plt.show()