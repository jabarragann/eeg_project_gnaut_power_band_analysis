"""
Dataset generator v4
* Multitaper spectrum
* Rejection of noisy epochs
"""


from pathlib import Path
import sys
sys.path.append(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis')
import PowerClassification.Utils.NetworkTraining as ut
import numpy as np
import pandas as pd
from itertools import product
import re
import yasa
import mne
from collections import defaultdict
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps


def splitDataIntoEpochs(raw, frameDuration, overlap):
    w1 = frameDuration
    sf = 250

    eTime = int(w1 / 2 * sf) + raw.first_samp
    events_array = []
    while eTime < raw.last_samp:
        events_array.append([eTime, 0, 1])
        eTime += sf * (w1 - overlap)

    events_array = np.array(events_array).astype(np.int)
    epochs = mne.Epochs(raw, events_array, tmin=-(w1 / 2), tmax=(w1 / 2))

    return epochs


#['low', 'Delta','Theta','Alpha','Beta', 'Gamma']
# EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
#                   "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
#                   "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
#                   "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]
# Power_coefficients = ['Delta','Theta','Alpha','Beta']
EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO3","PO4","OZ"]
Power_coefficients = ['Theta','Alpha','Beta']

newColumnNames = [x+'-'+y for x,y in product(EEG_channels, Power_coefficients)] + ['Label']
print(newColumnNames)

#Global variables
sf = 250

users = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']
# users = ['UI01','UI02']

# windowSize = [10, 20, 30]
windowSize = [2,10,20]

rawDataPath = Path('C:\\Users\\asus\\OneDrive - purdue.edu\\RealtimeProject\\Experiment1-Pilot-Final')
# rawDataPath = Path(r'C:\Users\asus\PycharmProjects\EEG-recording-lsl\Bleed-real-time-test\dataset')
data_preprocess = 'pyprep'

dstPath = Path('./data/') / 'feature-selection-multitaper-drop-pyprep'

#Sessions black list
black_list = {'UI01':['1','6','3','7'],
              'UI02':['7','4','2'],
              'UI03':['2'],
              'UI04':['4'],
              'UI05':['3'],
              'UI06':['2'],
              'UI07':[''],
              'UI08':['']}
useBlackList = False

def multitaper_bandpower(data,sf, eeg_channels, bands=None, relative=True):
    if bands is None:
        bands=[(0.5, 4, 'Delta'),(4, 8, 'Theta'),
                 (8, 12, 'Alpha'), (12, 30, 'Beta'),
                 (30, 50, 'Gamma')]

    psd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0, n_jobs=4)

    freq_res = freqs[1] - freqs[0]  # Frequency resolution

    bandpower_df = pd.DataFrame(columns=eeg_channels)
    for ch_idx,ch  in enumerate(eeg_channels):
        for lf,hf,bn in bands:
            # Find index of band in frequency vector
            idx_band = np.logical_and(freqs >= lf, freqs <= hf)
            # Integral approximation of the spectrum using parabola (Simpson's rule)
            bp = simps(psd[ch_idx,idx_band], dx=freq_res)
            bandpower_df.loc[bn, ch] = bp

        if relative:
            bandpower_df[ch] = bandpower_df[ch] / simps(psd[ch_idx,:], dx=freq_res)

    return bandpower_df.transpose

def multitaper_bandpower(data,sf, eeg_channels, bands=None, relative=True,n_jobs=5):
    if bands is None:
        bands=[(0.5, 4, 'Delta'),(4, 8, 'Theta'),
                 (8, 12, 'Alpha'), (12, 30, 'Beta'),
                 (30, 50, 'Gamma')]

    psd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0, n_jobs=n_jobs)

    freq_res = freqs[1] - freqs[0]  # Frequency resolution

    bandpower_df = pd.DataFrame(columns=eeg_channels).astype(float)
    for ch_idx,ch  in enumerate(eeg_channels):
        for lf,hf,bn in bands:
            # Find index of band in frequency vector
            idx_band = np.logical_and(freqs >= lf, freqs <= hf)
            # Integral approximation of the spectrum using parabola (Simpson's rule)
            bp = simps(psd[ch_idx,idx_band], dx=freq_res)
            bandpower_df.loc[bn, ch] = bp

        total_power = simps(psd[ch_idx, :], dx=freq_res)
        if relative:
            bandpower_df[ch] = bandpower_df[ch] / total_power
        bandpower_df.loc['TotalAbsPow', ch] = total_power

    return bandpower_df.transpose()


#Reduce to only 4 minutes of data
def calculatePowerBand(epoched_data, data_label, window, trial):
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
        bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 50, 'Gamma')]
        bd = multitaper_bandpower(data,sf=sf,eeg_channels=EEG_channels,bands=bands, relative=True)
        # bd = yasa.bandpower(data, sf=sf, ch_names=EEG_channels, win_sec=win_sec,
        #                     bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'),
        #                            (12, 30, 'Beta'), (30, 50, 'Gamma')],
        #                     bandpass=False, relative=True)

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


def main():

    drop_log_per_u = defaultdict(int)
    total_epochs_log = defaultdict(int)

    mne.set_log_level("WARNING")
    utilities = ut.Utils()

    # Create Directory where all the data is going to be stored
    utilities.makeDir(dstPath)

    for w1 in windowSize:
        utilities.makeDir(dstPath / '{:02d}s'.format(w1))

        # Check all the raw files and create a file with the specified data file
        for file in rawDataPath.rglob(('*pyprep.edf')):
            # Rename files --> remove identifiers
            uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
            session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
            trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
            task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
            preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

            # Only use files from a specific preprocess
            if uid in users and preprocess == data_preprocess and task != 'Baseline':
                if useBlackList and session in black_list[uid]:
                    print("Black listed, ", file.name)
                else:
                    dstPathFinal = dstPath / '{:02d}s/{:}'.format(w1, uid)

                    if not Path.exists(dstPathFinal):
                        utilities.makeDir(dstPathFinal)

                    # read file
                    raw = mne.io.read_raw_edf(file)
                    raw.pick(EEG_channels)
                    raw.load_data()
                    raw.filter(0.5, 30)

                    # Split data into epochs
                    epochs = splitDataIntoEpochs(raw, w1, 0.0)
                    orig_size = epochs.get_data().shape[0]
                    epochs.drop_bad(reject={"eeg":7.277617667499599e-05})
                    reduced_size = epochs.get_data().shape[0]
                    drop_log_per_u[uid] += orig_size - reduced_size
                    total_epochs_log[uid] += orig_size

                    # Label
                    label = -1
                    assert task in ['pegInversion', 'pegNormal', 'Low', 'High'], \
                                    '{:} is not recognized as a label'.format(task)
                    if task == 'pegInversion' or task == 'High':
                        label = 1.0
                    elif task == 'pegNormal' or task == 'Low':
                        label = 0.0

                    # calculate power bands
                    print(file.name)
                    print(label)
                    powerBandFile = calculatePowerBand(epochs, label, w1, trial)

                    pf = dstPathFinal / '{:}_S{:}_T{:}_pow.txt'.format(uid, session, trial)
                    powerBandFile.to_csv(pf, sep=',')

    print(drop_log_per_u)
    print(total_epochs_log)

if __name__ == '__main__':
    main()