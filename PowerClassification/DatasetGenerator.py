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

#Channels PO7 and PO8 are not included
EEG_channels = [
                    "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                    "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                    "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                    "PZ","P4","P8","PO3","PO4","OZ"]

Power_coefficients = ['Low','Delta','Theta','Alpha','Beta', 'Gamma']

newColumnNames = [x+'-'+y for x,y in product(EEG_channels, Power_coefficients)] + ['Label']
print(newColumnNames)

#Global variables
# users = ['juan','jackie','ryan','jhony']
users = ['juanBaseline', 'juan']
windowSize = [60]
dataPath = Path('./data/')
rawDataPath = Path('./data/raw_data')
dstPath = dataPath / 'DifferentWindowSizeData'
sf = 250


def calculatePowerBand(windowArray, df ):
    counter=0
    dataDict = {}
    for i in range(windowArray.shape[0]):

        v1 =  windowArray[i]

        data = df[int(v1[1]):int(v1[2]+1)]

        data2 = data[EEG_channels].transpose().values

        # print(counter, data2.shape[1], data2.shape[0])

        if data2.shape[1]>data2.shape[0]*2:
            # (0.0, 0.5, 'Low'), (0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),(12, 30, 'Beta'), (30, 50, 'Gamma')
            # Calculate bandpower
            bd = yasa.bandpower(data2, sf=sf, ch_names=EEG_channels, win_sec=4,
                                bands=[(0.0, 0.5, 'Low'), (0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                       (12, 30, 'Beta'), (30, 50, 'Gamma')])
            # Reshape coefficients into a single row vector
            bd = bd[Power_coefficients].values.reshape(1, -1)

            # Create row name, label and add to data dict
            rowName = 'T' + str(trial) + '_' + str(counter)
            meanLabel = np.mean(data['label'].values)
            if meanLabel > 7.5:
                label = 1
            elif 7.5 > meanLabel > 1:
                label = 0
            else:
                label = -1
            bd = np.concatenate((bd, np.array([label]).reshape(1, -1)), axis=1)
            dataDict[rowName] = np.squeeze(bd)
            #Update counter
            counter+=1

    powerBandDataset = pd.DataFrame.from_dict(dataDict, orient='index', columns=newColumnNames)

    return powerBandDataset

if __name__ == '__main__':

    utilities = ut.Utils()

    #Create Directory where all the data is going to be stored
    utilities.makeDir(dstPath)

    for w1 in windowSize:

        utilities.makeDir(dstPath/'{:02d}s'.format(w1))

        #Check all the raw files and create a file with the specified data file
        for f2 in rawDataPath.rglob(('*.txt')):
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

            #Initial values
            result = df.loc[(df['markerValue'] == 'not_active') &
                             (df['5SecondWindow'] == 'WindowStart')]
            startIdx = result.index.values[0]
            startTime = result['COMPUTER_TIME'].values[0]

            windowCounter = 0
            windowArray = []
            while True:
                #Get end Idx
                result = df.loc[df['COMPUTER_TIME'] > (startTime+w1)]

                #Check if end was reached
                if result.shape[0] <= 1:
                    endIdx = df.index.values[-1]
                    endTime = df['COMPUTER_TIME'].values[-1]
                    t = [windowCounter, startIdx, endIdx, endTime-startTime]
                    windowArray.append(t)
                    break

                endIdx = result.index.values[0]
                endTime = result['COMPUTER_TIME'].values[0]
                t = [windowCounter,startIdx,endIdx, endTime-startTime]
                windowArray.append(t)

                #Update values
                windowCounter += 1
                startIdx = endIdx + 1
                startTime = result['COMPUTER_TIME'].values[1]

            windowArray =  np.array(windowArray)
            powerBandFile = calculatePowerBand(windowArray,df)

            pf = dstPathFinal / '{:}_S{:d}_T{:}_pow.txt'.format(u1,session,trial)
            powerBandFile.to_csv(pf, sep=',')

            print(pf)

        utilities.sendSMS('The dataset creation finished')
