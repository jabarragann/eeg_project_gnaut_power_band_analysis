import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)

##Software is removing last epoch of data
##Solution create events manually
mne.set_log_level("WARNING")

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

#Read eeg file
file = Path('./../data/juan_S3_T2_epoc_pyprep.edf')
raw = mne.io.read_raw_edf(file)

#Rename Channel and set montage (3d electrode location)
mne.rename_channels(raw.info, renameChannels)
raw = raw.set_montage('standard_1020')
raw.set_channel_types({'Fp1':'eog','Fp2':'eog'})

eog_evoked = create_eog_epochs(raw, picks=["eeg"])
eog_evoked.plot_image(combine='mean', show=False)
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.average().plot_joint(show=False)
eog_evoked.plot()
# raw.plot()
plt.show()