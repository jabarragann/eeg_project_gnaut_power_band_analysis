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
                  "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]


#Global variables
users = ['juan','jackie','ryan','jhony','karuna', 'santy']
# users = ['karuna']
epochSize = [10,20,30]
dataPath = Path('./data/')
rawDataPath = Path('./../PowerClassification/data/raw_data_pyprep')
dstPath = dataPath / 'DifferentWindowSizeData_pyprep'
sf = 250


if __name__ == '__main__':

    utilities = ut.Utils()

    #Create Directory where all the data is going to be stored
    utilities.makeDir(dstPath)

    #Set log only to warnings
    mne.set_log_level("WARNING")

    for w1 in epochSize:

        utilities.makeDir(dstPath/'{:02d}s'.format(w1))

        #Check all the raw files and create a file with the transformed data
        for f2 in rawDataPath.rglob(('*.txt')):
            if f2.parent.name in users:
                dstPathFinal = dstPath/'{:02d}s/{:}'.format(w1,f2.parent.name)
                u1 = f2.parent.name
                if not Path.exists( dstPathFinal ):
                    utilities.makeDir(dstPathFinal)

                # Get trial and session. Open data with pandas
                trial = re.findall("S[0-9]_T[0-9]", f2.name)
                trial = int(trial[0][-1])
                session = re.findall("S[0-9]_T[0-9]", f2.name)
                session = int(session[0][-4])
                df = pd.read_csv(f2, sep=',')
                data = df[EEG_channels+["label"]].values.transpose()

                #Load Data to MNE
                data = data / 1e6
                ch_names = EEG_channels + ["labelCh"]
                ch_types = ["eeg"] * len(EEG_channels) + ["stim"]
                info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
                raw = mne.io.RawArray(data, info)

                #Split data into epochs
                totalPoints = data.shape[1]
                nperE = sf * w1 #Number of samples per Epoch

                eTime = int(w1 / 2 * sf)
                events_array = [[eTime,0,1]]
                while eTime < totalPoints:
                    eTime += sf*w1
                    events_array.append([eTime,0,1])
                events_array = np.array(events_array)

                epochs = mne.Epochs(raw, events_array, tmin=-(w1/2-0.02*w1), tmax=(w1/2-0.02*w1))
                epochs.load_data()
                epochs = epochs.filter(0.5, 30)

                # #Plot Epochs
                # epochs = epochs[:1]
                # epochs.plot()
                # epochs[0].plot_psd(picks='eeg')
                # plt.show()

                #Get data
                epochedData  = epochs.get_data(picks=['eeg']) * 1e6 #Convert back to micro volts
                epochedData = epochedData.reshape(epochedData.shape[0],1,32,-1)
                epochedLabel = epochs.get_data(picks=['stim']) * 1e6
                epochedLabel = epochedLabel.mean(axis=2).squeeze()
                epochedLabel = np.array(list(map(lambda x: 1 if x > 7.5 else 0,epochedLabel))) #Change labels to 1 and 0

                #Save epochs
                pf = dstPathFinal / '{:}_S{:d}_T{:}_time.pickle'.format(u1,session,trial)
                with open(pf,'wb') as f:
                    dataDict = {'X':epochedData,'y':epochedLabel}
                    pickle.dump(dataDict, f)

                print(pf.name)
