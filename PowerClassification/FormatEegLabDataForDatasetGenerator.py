from pathlib import Path
import sys
sys.path.append(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis')
#Import libraries
import PowerClassification.Utils.NetworkTraining as ut
from itertools import product
import re
import pandas as pd

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
users = ['juan','jackie','ryan','jhony']
windowSize = [10,15,20,25,30,40]
dataPath = Path('./data/')
rawDataPath = Path('./data/raw_data')
rawEegLabPath = Path('./data/raw_data_eeglab')
dstPath = Path('./data/raw_data_eeglab_formatted')
sf = 250

if __name__ == '__main__':

    utilities = ut.Utils()

    #Create Directory where all the data is going to be stored
    utilities.makeDir(dstPath)

    #Check all the raw files and create a file with the specified data file
    for eeglabFile in rawEegLabPath.rglob(('*.txt')):
        fileName = eeglabFile.name

        user = re.findall('[A-Za-z]*(?=_S)',fileName)[0]

        t = dstPath / "{:}".format(user)
        if not t.exists():
            t.mkdir(parents=True)

        session = int(re.findall('[0-9](?=_T)',fileName)[0])
        trial = re.findall('(?<=[0-9]_T)[0-9]',fileName)
        if len(trial) > 0:
            trial = int(trial[0])
            print(fileName, user, session, trial)

            eegLabFrame = pd.read_csv(eeglabFile,sep='\t')

            rawP = rawDataPath / "{:}".format(user) / "{:}_S{:d}_T{:d}_epoc.txt".format(user,session,trial)
            rawFrame = pd.read_csv(rawP, sep=',')

            eegLabFrame.columns = [s.upper() for s in eegLabFrame.columns]
            eegLabFrame["COMPUTER_TIME"] = rawFrame["COMPUTER_TIME"]
            eegLabFrame["isThereMarker"] = rawFrame["isThereMarker"]
            eegLabFrame["markerValue"] = rawFrame["markerValue"]
            eegLabFrame["label"] = rawFrame["label"]
            eegLabFrame["5SecondWindow"] = rawFrame["5SecondWindow"]
            eegLabFrame["5SecondWindow"] = rawFrame["5SecondWindow"]


            eegLabFrame.to_csv(t/ "{:}_S{:d}_T{:d}_ica.txt".format(user,session,trial), index=False)


