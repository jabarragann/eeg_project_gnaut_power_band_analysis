import numpy as np
import pandas as pd
import os
from itertools import product
import re
import yasa

#Channels PO7 and PO8 are not included
EEG_channels_EEGLAB = [
                    "Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4",
                    "F8","FC5","FC1","FC2","FC6","T7","C3","Cz",
                    "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                    "Pz","P4","P8","PO3","PO4","Oz"]

EEG_channels = [
                    "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                    "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                    "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                    "PZ","P4","P8","PO3","PO4","OZ"]

Power_coefficients = ['Low','Delta','Theta','Alpha','Beta']

if __name__ == '__main__':

    #Sample frequency
    sf = 250

    user = 'jhony'
    dataPath = './raw_data_eeglab/{}/'.format(user)
    dataPathRaw = './raw_data/{}/'.format(user)

    files = os.listdir(dataPath)
    newColumnNames = [x+'-'+y for x,y in product(EEG_channels, Power_coefficients)] + ['Label']
    print(newColumnNames)

    for file in files:
        if file != 'empty.py' and len(re.findall('csv|Ticaed',file)) == 0 :

            trial = re.findall("S[0-9]_T[0-9]", file)
            trial = int(trial[0][-1])
            session = re.findall("S[0-9]", file)
            session = int(session[0][-1])
            print(file, session, trial)

            eeglabData = pd.read_csv(dataPath + file, sep='\t')

            #Read raw file to get timestamps and label
            rawData = pd.read_csv(dataPathRaw + '{}_S{}_T{}_epoc.txt'.format(user,session,trial), sep=',')

            #Remove data points before the starting signal and after the ending signal.
            startIdx = rawData[rawData['markerValue'] == 'started'].index.values[0]
            finishIdx = rawData[rawData['markerValue'] == 'finished'].index.values[0]
            rawData = rawData[startIdx:finishIdx+1]

            eegData = eeglabData[EEG_channels_EEGLAB].values

            eegLabel = rawData["label"].values
            eegEvents = rawData["markerValue"].values
            eegWindows = rawData["5SecondWindow"].values

            print(rawData.shape[0],eegData.shape[0])
            #Get Initial index
            for i in range(0, len(eegEvents)):
                if eegEvents[i] == 'started':
                    for j in range(i, len(eegEvents)):
                        if eegEvents[j] == 'active' or eegEvents[j] == 'not_active':
                            initialIdx = j
                            break
                    break

            counter = 0
            dataDict = {}

            for j in range(initialIdx, len(eegEvents)):
                if eegWindows[j + 1] == 'WindowStart' or eegEvents[j + 1] == 'finished':

                    # print("Sample {:d}".format(counter))
                    data = eegData[initialIdx:j + 1]
                    data = data.transpose()
                    # print(data.shape[1])

                    if data.shape[1]>1240:
                        #(0.0, 0.5, 'Low'), (0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),(12, 30, 'Beta'), (30, 50, 'Gamma')
                        #Calculate bandpower
                        bd = yasa.bandpower(data, sf=sf, ch_names=EEG_channels, win_sec=4,
                                            bands=[(0.0, 0.5, 'Low'), (0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                   (12, 30, 'Beta'), (30, 50, 'Gamma')])
                        #Reshape coefficients into a single row vector
                        bd = bd[Power_coefficients].values.reshape(1, -1)

                        #Create row name, label and add to data dict
                        rowName = 'T' + str(trial) + '_' + str(counter)
                        label = 1 if np.mean(eegLabel[initialIdx:j]) > 7.5 else 0
                        bd = np.concatenate((bd,np.array([label]).reshape(1, -1) ), axis=1)
                        dataDict[rowName] = np.squeeze(bd)

                    else:
                        print("Sample {:d} not included".format(counter))

                    initialIdx = j + 1
                    counter += 1

                if eegEvents[j + 1] == 'finished':
                    print("Number of samples {:d}".format(counter))

                    powerBandDataset = pd.DataFrame.from_dict(dataDict, orient='index', columns=newColumnNames)
                    powerBandDataset.to_csv('./data/usersEEGLab/{:}/'.format(user)+ file[:-16]+'_pow_eeglab.txt', sep=',')
                    break
