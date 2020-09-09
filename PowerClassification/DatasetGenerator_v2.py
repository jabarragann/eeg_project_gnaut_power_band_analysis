'''
The following script will allow to generate processed datasets ready for classification
with different windows sizes. Until now we have used only windows of 5 seconds to make
classifications; however, it is not know if this is the optimal value. This script will
attempt to solve this question.
'''


from pathlib import Path
import sys
sys.path.append(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis')
#Import libraries
import PowerClassification.Utils.NetworkTraining as ut
import numpy as np
import pandas as pd
import os
from itertools import product
import re
import yasa
import mne

#['low', 'Delta','Theta','Alpha','Beta', 'Gamma']
#Channels PO7 and PO8 are not included
EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]

Power_coefficients = ['Delta','Theta','Alpha','Beta']

newColumnNames = [x+'-'+y for x,y in product(EEG_channels, Power_coefficients)] + ['Label']
print(newColumnNames)

#Global variables
users = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']

# windowSize = [10, 20, 30]
windowSize = [2, 5, 10, 20, 30]
rawDataPath = Path('C:\\Users\\asus\\OneDrive - purdue.edu\\RealtimeProject\\Experiment1-Pilot')
data_preprocess = 'pyprep'
dstPath = Path('./data/') / 'de-identified-{:}-dataset-reduced-critically-exp'.format(data_preprocess)
sf = 250

#Sessions black list
black_list = {'UI01':['1','6','3','7'],
              'UI02':['7','4','2'],
              'UI03':['2'],
              'UI04':['4'],
              'UI05':['3'],
              'UI06':['2'],
              'UI07':[''],
              'UI08':[''],}

#Reduce to only 4 minutes of data
def calculatePowerBand(epoched_data, data_label, window):
    counter=0
    dataDict = {}

    epoched_data.load_data()


    if window < 4:
        win_sec = window * 0.95
    else:
        win_sec = 4

    for i in range(len(epoched_data)):

        data  =  epoched_data[i]
        data = data.get_data().squeeze()
        data  *= 1e6

        # (0.0, 0.5, 'Low'), (0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),(12, 30, 'Beta'), (30, 50, 'Gamma')
        # Calculate bandpower
        bd = yasa.bandpower(data, sf=sf, ch_names=EEG_channels, win_sec=win_sec,
                            bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                   (12, 30, 'Beta'), (30, 50, 'Gamma')])

        # Reshape coefficients into a single row vector
        bd = bd[Power_coefficients].values.reshape(1, -1)

        # Create row name, label and add to data dict
        rowName = 'T' + str(trial) + '_' + str(counter)

        bd = np.concatenate((bd, np.array([data_label]).reshape(1, -1)), axis=1)
        dataDict[rowName] = np.squeeze(bd)
        #Update counter
        counter+=1

    powerBandDataset = pd.DataFrame.from_dict(dataDict, orient='index', columns=newColumnNames)

    return powerBandDataset

if __name__ == '__main__':
    mne.set_log_level("WARNING")
    utilities = ut.Utils()

    #Create Directory where all the data is going to be stored
    utilities.makeDir(dstPath)

    for w1 in windowSize:
        utilities.makeDir(dstPath/'{:02d}s'.format(w1))

        #Check all the raw files and create a file with the specified data file
        for file in rawDataPath.rglob(('*.edf')):
            # Rename files --> remove identifiers
            uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
            session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
            trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
            task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
            preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

            #Only use files from a specific preprocess
            if uid in users and preprocess == data_preprocess and task !='Baseline':
                if session in black_list[uid]:
                    print("Black listed, ", file.name)
                else:
                    dstPathFinal = dstPath/'{:02d}s/{:}'.format(w1,uid)

                    if not Path.exists( dstPathFinal ):
                        utilities.makeDir(dstPathFinal)

                    #read file
                    raw = mne.io.read_raw_edf(file)
                    #Filter data
                    raw.load_data()
                    raw.filter(0.5, 30)

                    #Reduce files to 4 minutes only
                    maxPoints = 60000 # 250hz * 5 min * 60 s
                    totalPoints = raw.get_data().shape[1]

                    if totalPoints >  maxPoints:
                        extraTime = totalPoints -  maxPoints #In samples
                        extraTime  = extraTime / 250 #In seconds
                        maxTime =    totalPoints /250 #In seconds

                        raw.crop(tmin=extraTime/2, tmax=maxTime - extraTime/2)
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
                    # epochs.load_data()
                    #Changed this instead of filtering the epochs to filter the entire signal.
                    # epochs = epochs.filter(0.5, 30)

                    #Label
                    assert task in ['pegInversion','pegNormal'], '{:} is not recognized as a label'.format(task)
                    if task == 'pegInversion':
                        label = 1
                    elif task == 'pegNormal':
                        label = 0

                    #calculate power bands
                    print(file.name)
                    powerBandFile = calculatePowerBand(epochs, label, w1)

                    pf = dstPathFinal / '{:}_S{:}_T{:}_pow.txt'.format(uid,session,trial)
                    powerBandFile.to_csv(pf, sep=',')

                    # print(pf)


