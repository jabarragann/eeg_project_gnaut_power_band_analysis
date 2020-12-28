"""
Script to find the optimal rejection threshold for epochs
"""
import sys
sys.path.append(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis')
from pathlib import Path
import numpy as np
from itertools import product
import re
import mne
from autoreject import get_rejection_threshold

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



def main():
    # ['low', 'Delta','Theta','Alpha','Beta', 'Gamma']
    EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                      "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                      "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                      "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]
    Power_coefficients = ['Delta','Theta','Alpha','Beta']
    newColumnNames = [x + '-' + y for x, y in product(EEG_channels, Power_coefficients)] + ['Label']

    # Global variables
    # users = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']
    users = ['UI05']
    windowSize = [1]
    rawDataPath = Path('C:\\Users\\asus\\OneDrive - purdue.edu\\RealtimeProject\\Experiment1-Pilot-Final')
    data_preprocess = 'pyprep'
    dstPath = Path('./data/') / 'feature-selection-pyprep'
    sf = 250

    total_list = []
    w1 = windowSize[0]
    for file in rawDataPath.rglob(('*pyprep.edf')):
        # Rename files --> remove identifiers
        uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
        session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
        trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
        task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
        preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

        #Only use files from a specific preprocess
        if uid in users and preprocess == data_preprocess and task !='Baseline':
            print(file.name)
            #read file
            raw = mne.io.read_raw_edf(file)
            raw.drop_channels(["PO8","PO7"])
            #Filter data
            raw.load_data()
            raw.filter(0.5, 30)

            total_list.append(splitDataIntoEpochs(raw,w1,0.5))

    epochs = mne.concatenate_epochs(total_list)
    # reject = get_rejection_threshold(epochs, decim=2)

    # print('The rejection dictionary is %s' % reject)

    size1 = epochs.get_data().shape
    x = 0
    epochs = epochs.drop_bad(reject= {'eeg': 7.277617667499599e-05})
    size2 = epochs.get_data().shape
    print(size1, size2)

if __name__ == '__main__':
    mne.set_log_level("WARNING")
    main()
