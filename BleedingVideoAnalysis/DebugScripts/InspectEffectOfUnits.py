"""
Check if the units affect the spectral coeffients calculation
Apparently there are no problems
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import yasa
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import re
import pandas as pd
import mne
from itertools import product
from pathlib import Path

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')

    return chName

Power_coefficients = ['Theta', 'Alpha', 'Beta']
EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO3","PO4","OZ"]
newColumnNames = [x+'-'+y for x,y in product(Power_coefficients,renameChannels(EEG_channels))]


def getBandPowerCoefficients(epoch_data):
    counter = 0
    dataDict = {}
    # epoch_data.load_data()
    win_sec =0.95
    sf = 250

    for i in range(len(epoch_data)):
        data = epoch_data[i]
        data = data.squeeze() #Remove additional
        data *= 1e6

        # Calculate bandpower
        bd = yasa.bandpower(data, sf=sf, ch_names=EEG_channels, win_sec=win_sec,
                            bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 40, 'Beta')])
        # Reshape coefficients into a single row vector with the format
        # [Fp1Theta,Fp2Theta,AF3Theta,.....,Fp1Alpha,Fp2Alpha,AF3Alpha,.....,Fp1Beta,Fp2Beta,AF3Beta,.....,]
        bandpower = bd[Power_coefficients].transpose()
        bandpower = bandpower.values.reshape(1, -1)
        # Create row name, label and add to data dict
        rowName = 'T' + str(i) + '_' + str(counter)
        dataDict[rowName] = np.squeeze(bandpower)
        # Update counter
        counter += 1

    powerBandDataset = pd.DataFrame.from_dict(dataDict, orient='index', columns=newColumnNames)

    return powerBandDataset



def main():
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data")
    srcPath = srcPath / r"TestsWithVideo\Eyes-open-close-test\T01"
    srcPath = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name)) > 0][0]
    print("loading eeg from {:}".format(srcPath.name))


    raw_array = pickle.load(open('../CheckPredictionPlot/array_of_raw.pickle', 'rb'))
    raw_array = np.array(raw_array)

    raw_array = raw_array.transpose((0,2,1))
    powerBand1 = getBandPowerCoefficients(raw_array)
    powerBand2 = getBandPowerCoefficients(raw_array * 1e-6)

    # fig, axes = plt.subplots(2, 1)
    # epochs = epochs.transpose((0,2,1))
    # raw_array = raw_array[1:,:,:]
    # axes[0].plot(epochs[10,:,0])
    # axes[0].plot(raw_array[10,:,0])
    # axes[1].plot(epochs[40, :, 0])
    # axes[1].plot(raw_array[40, :, 0])

    plt.show()

if __name__ == "__main__":
    main()

