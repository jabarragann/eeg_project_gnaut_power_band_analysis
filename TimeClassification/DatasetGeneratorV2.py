'''
Create a dataset directly from edf files saved in one drive.
'''

from pathlib import Path
import sys
sys.path.append(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis')
#Import libraries
import PowerClassification.Utils.NetworkTraining as ut
import TimeClassification.Utils.NetworkTraining
import numpy as np
import pandas as pd
import re

import mne

import pickle

##Time classification EEG dataset generator Script

EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO3","PO4","OZ"]


#Global variables
# users = ['juan','jackie','ryan','jhony','karuna', 'santy']
users = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']


epochSize = [10]
data_preprocess = 'pyprep'
dstPath = Path('./data/') / 'de-identified-{:}-dataset-reduced'.format(data_preprocess)
rawDataPath = Path('C:\\Users\\asus\\OneDrive - purdue.edu\\RealtimeProject\\Experiment1-Pilot')

sf = 250

#Sessions black list
black_list = {'UI01':['1'],
              'UI02':['7'],
              'UI03':['2'],
              'UI04':['4'],
              'UI05':['3'],
              'UI06':['2'],
              'UI07':[''],
              'UI08':[''],}


if __name__ == '__main__':

    utilities = ut.Utils()

    #Create Directory where all the data is going to be stored
    utilities.makeDir(dstPath)

    #Set log only to warnings
    mne.set_log_level("WARNING")

    for w1 in epochSize:
        utilities.makeDir(dstPath/'{:02d}s'.format(w1))

        # Check all the raw files and create a file with the specified data file
        for file in rawDataPath.rglob(('*.edf')):
            # Rename files --> remove identifiers
            uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
            session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
            trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
            task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
            preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

            # Only use files from a specific preprocess
            if uid in users and preprocess == data_preprocess and task != 'Baseline':
                if session in black_list[uid]:
                    print("Black listed, ", file.name)
                else:
                    dstPathFinal = dstPath / '{:02d}s/{:}'.format(w1, uid)

                    if not Path.exists(dstPathFinal):
                        utilities.makeDir(dstPathFinal)

                    # read file
                    raw = mne.io.read_raw_edf(file)

                    # Reduce files to 4 minutes only
                    maxPoints = 60000  # 250hz * 5 min * 60 s
                    totalPoints = raw.get_data().shape[1]

                    if totalPoints > maxPoints:
                        extraTime = totalPoints - maxPoints  # In samples
                        extraTime = extraTime / 250  # In seconds
                        maxTime = totalPoints / 250  # In seconds

                        raw.crop(tmin=extraTime / 2, tmax=maxTime - extraTime / 2)

                    # Split data into epochs
                    totalPoints = raw.get_data().shape[1]
                    nperE = sf * w1  # Number of samples per Epoch

                    eTime = int(w1 / 2 * sf)
                    events_array = []

                    while eTime < totalPoints:
                        events_array.append([eTime, 0, 1])
                        eTime += sf * w1

                    events_array = np.array(events_array)

                    epochs = mne.Epochs(raw, events_array, tmin=-(w1 / 2 - 0.02 * w1), tmax=(w1 / 2 - 0.02 * w1))
                    epochs.load_data()
                    epochs = epochs.filter(0.5, 30)

                    #Downsample data from 250 to 125hz for faster processing.
                    epochs_resampled = epochs.copy().resample(125, npad='auto')

                    #Get Label
                    assert task in ['pegInversion', 'pegNormal'], '{:} is not recognized as a label'.format(task)
                    if task == 'pegInversion':
                        label = 1
                    elif task == 'pegNormal':
                        label = 0

                    #Get data
                    epochedData  = epochs_resampled.get_data(picks=['eeg']) * 1e6 #Convert back to micro volts
                    epochedData = epochedData.reshape(epochedData.shape[0],1,32,-1)
                    epochedLabels = np.zeros(epochedData.shape[0]) + label

                    # #Save epochs
                    pf = dstPathFinal / '{:}_S{:}_T{:}_time.pickle'.format(uid,session,trial)
                    with open(pf,'wb') as f:
                        dataDict = {'X':epochedData,'y':epochedLabels}
                        pickle.dump(dataDict, f)

                    print(pf.name)
