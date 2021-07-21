import sys
sys.path.append('./../')

import pickle
from pathlib import Path
import EyeTrackerClassification.EyeTrackerUtils as etu
from scipy.signal import welch
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yasa
import numpy as np
import re

EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO3","PO4"]

# EEG_channels = ["FZ","PZ","CZ","C3","C4","CP1","CP2"]

def get_file_information(file):
    uid = re.findall('.+(?=_S[0-9]+_T[0-9]+_)', file.name)[0]
    session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{2}_)', file.name)[0])
    trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]+(?=_)', file.name)[0])
    task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_eye)', file.name)[0]

    info_dict = dict(uid=uid,session=session,trial=trial,task=task,session_folder=file.parent.name)
    return info_dict


def split_eeg_into_epochs(eeg_df,ts_events, window_size=15):
    eeg_epochs = []
    eeg_ts = []
    for idx, ts in enumerate(ts_events):
        e = eeg_df.loc[ (eeg_df['LSL_TIME']>ts-window_size/2) & (eeg_df['LSL_TIME']<ts+window_size/2)  ]
        if e.shape[0]>1000:
            eeg_epochs.append(e[EEG_channels].values[:1000,:])
            eeg_ts.append(ts)
        else:
            print("epoch {:d} has not enough data".format(idx), e.shape)

    eeg_epochs = np.array(eeg_epochs)

    return eeg_epochs, eeg_ts

def main():
    window_size = 15
    eye_tracker_path = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Realtime-Project-IU-experiments\eyetracker\participant07\S01')
    eeg_path = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Realtime-Project-IU-experiments\txt\participant07\S01')
    dst_path = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Realtime-Project-IU-experiments\fusefeatures')

    labels = ['NeedlePassing', 'BloodNeedle']

    files_dict_paths = dict(etu.search_files_on_path_2(eye_tracker_path))

    files_dict = etu.merge_files(files_dict_paths, labels)

    for key in files_dict.keys():
        eye_df = files_dict[key]
        information = get_file_information(files_dict_paths[key]['left'])

        eye_x, eye_y, ts_events = etu.get_data_single_file(eye_df, window_size=window_size)
        print("eye tracker shape", eye_x.shape)

        #Load corresponding eeg file
        eeg_file = '{:}_S{:02d}_T{:02d}_{:}_raw.txt'.format(information['uid'], information['session'],information['trial'], information['task'])
        print("load ", eeg_file)
        eeg_file = pd.read_csv(eeg_path/eeg_file)
        # eeg_file = eeg_file[EEG_channels]

        #Split eeg into epochs with ts_events (ts-12.5, ts+12.5)
        eeg_epochs,eeg_ts = split_eeg_into_epochs(eeg_file, ts_events,window_size=window_size) #Numpy array (epochs, channels, samples)
        eeg_epochs = eeg_epochs.transpose([0,2,1])
        #Calculate PSD
        sf=250
        win = int(4 * sf)  # Window size is set to 4 seconds
        freqs, psd = welch(eeg_epochs, sf, nperseg=win, axis=-1)

        #Calculate bandpower
        # bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
        #          (12, 16, 'Sigma'), (16, 30, 'Beta')]
        bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta')]
        bands_names = ['Theta','Alpha','Sigma','Beta']
        # Calculate the bandpower on 3-D PSD array
        bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=False)
        bandpower = bandpower.transpose([1,2,0])
        bandpower = bandpower.mean(axis=1)
        bandpower = pd.DataFrame(bandpower, columns=bands_names)
        #Concatenate with eye tracker
        eye_cols = list(eye_x.columns.values)
        fuse_df = pd.DataFrame(np.hstack((eye_x.values[:bandpower.shape[0],:],bandpower)),columns=(eye_cols+bands_names))
        fuse_df['LSL_TIME'] = ts_events[:bandpower.shape[0]]

        #Save fused features

        dst = dst_path / information['uid'] / information['session_folder']
        if not dst.exists():
            dst.mkdir(parents=True)
        fuse_df.to_csv(dst/'{:}_S{:02d}_T{:02d}_{:}_fuse.txt'.format(information['uid'], information['session'],information['trial'], information['task']))


    x = 0

if __name__ == "__main__":
    main()