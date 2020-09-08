from tensorflow.keras.utils import to_categorical
from pathlib import Path
import numpy as np

def separateIntoTrainValTest(data):
    trainX = []
    trainY = []
    valX = []
    valY = []
    testX = []
    testY = []

    for key in data.keys():
        if key == '5': #Testing samples
            testX.append(data[key]['X'])
            testY.append(data[key]['y'])
        elif key == '4': #Validation samples
            valX.append(data[key]['X'])
            valY.append(data[key]['y'])
        else: #training samples
            trainX.append(data[key]['X'])
            trainY.append(data[key]['y'])

    trainX = np.concatenate(trainX)
    trainY = to_categorical(np.concatenate(trainY))
    valX = np.concatenate(valX)
    valY = to_categorical(np.concatenate(valY))
    testX = np.concatenate(testX)
    testY = to_categorical(np.concatenate(testY))

    return trainX,trainY, valX,valY, testX,testY

#Global variables
EEG_CHANNELS = [
            "FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
            "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
            "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
            "PZ", "P4", "P8", "PO3","PO7", "PO4", "PO8", "OZ"]
POWER_COEFFICIENTS = ['Delta', 'Theta', 'Alpha', 'Beta']
DATA_PATH_10 = Path('./../data/de-identified-pyprep-dataset-reduced/{:02d}s/'.format(10)).resolve()
DATA_PATH_20 = Path('./../data/de-identified-pyprep-dataset-reduced/{:02d}s/'.format(20)).resolve()
DATA_PATH_30 = Path('./../data/de-identified-pyprep-dataset-reduced/{:02d}s/'.format(30)).resolve()
DATA_DICT = {"data_10":DATA_PATH_10,"data_20":DATA_PATH_20,"data_30":DATA_PATH_30}
