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
raw = mne.io.read_raw_edf(file)

#Rename Channel
mne.rename_channels(raw.info,renameChannels)

#Set montage (3d electrode location)
raw = raw.set_montage('standard_1020')

#Create events every 20 seconds
events_array = mne.make_fixed_length_events(raw, start=10, stop=None, duration=20)
events_array = np.vstack((events_array , [67500+5000,0,1]))
# print(events_array)

# scalings = {'eeg': 'auto'}
# raw.plot(n_channels=32, scalings=scalings, title='Edf sample', show=True, block=True,
#          events=events_array,event_color={1:'r'}, duration=20.0)

raw.plot_sensors(kind='topomap', ch_type='all', show_names=True)
plt.show()