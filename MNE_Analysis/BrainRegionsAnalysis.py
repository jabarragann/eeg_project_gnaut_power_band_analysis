import re
import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

# Check all the raw files and create a file with the specified data file
rawDataPath =  Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiment1-Pilot\UI06\pyprep_edf').resolve()
lowTaskEpochs  = []
highTaskEpochs = []

#Sessions black list
black_list = {'UI01':['1','6','3','7'],
              'UI02':['7','4','2'],
              'UI03':['2'],
              'UI04':['4'],
              'UI05':['3'],
              'UI06':['2'],
              'UI07':[''],
              'UI08':[''],}
if __name__ == '__main__':

    for file in rawDataPath.rglob(('*.edf')):


        # Rename files --> remove identifiers
        uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
        session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
        trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
        task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
        preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

        if session in black_list:
            continue

        #Load and set montage locations
        raw = mne.io.read_raw_edf(file)
        mne.rename_channels(raw.info, renameChannels)
        raw = raw.set_montage('standard_1020')

        # Create events every 20 seconds and epoch data
        events_array = mne.make_fixed_length_events(raw, start=10, stop=None, duration=10)
        # events_array = np.vstack((events_array, [67500 + 5000, 0, 1]))
        epochs = mne.Epochs(raw, events_array, tmin=-9.5, tmax=9.5)

        if task == 'pegNormal':
            lowTaskEpochs.append(epochs)
        elif task == 'pegInversion':
            highTaskEpochs.append(epochs)


        print(uid, session, trial ,task , preprocess)

    #Concatenate epochs
    lowTaskEpochs = mne.concatenate_epochs(lowTaskEpochs)
    lowTaskEpochs.load_data()

    highTaskEpochs = mne.concatenate_epochs(highTaskEpochs)
    highTaskEpochs.load_data()

    #Frequency spatial distributions
    bands_list = [(0.5, 4, 'Delta-low'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta') ]
    low_fig = lowTaskEpochs.plot_psd_topomap(bands=bands_list ,ch_type='eeg', normalize=True, show=False)
    bands_list = [(0.5, 4, 'Delta-high'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta') ]
    high_fig = highTaskEpochs.plot_psd_topomap(bands=bands_list ,ch_type='eeg', normalize=True, show =False)

    plt.show()