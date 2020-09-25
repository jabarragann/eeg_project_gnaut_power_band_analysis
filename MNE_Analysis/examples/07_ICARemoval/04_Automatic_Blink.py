"""
Removing blinkings automatically by using the frontal channels as virtual Eog channels.

"""

import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)


mne.set_log_level("WARNING")

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

#Read eeg file
# file = Path('./../data/juan_S3_T2_epoc_pyprep.edf')
# file = Path('./../data/jackie_S3_T3_epoc_pyprep.edf')
file = Path('./../data/ryan_S3_T3_epoc_pyprep.edf')
raw = mne.io.read_raw_edf(file)

#Rename Channel and set montage (3d electrode location)
mne.rename_channels(raw.info, renameChannels)
raw = raw.set_montage('standard_1020')

#Setting frontal channels as EOG channels for blink detection
raw.set_channel_types({'Fp1':'eog'})

#Filter data for calculating ICA components
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)

ica = ICA(n_components=20, random_state=97)
ica.fit(filt_raw)

ica.exclude = []

# find which ICs match the ECG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw, )
ica.exclude = eog_indices

print("Excluding the following ICA components: ", eog_indices)

ica.plot_scores(eog_scores) # barplot of ICA component "EOG match" scores
ica.plot_properties(raw, picks=eog_indices) # plot diagnostics

