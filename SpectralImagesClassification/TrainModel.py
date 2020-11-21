#Misterious command that makes everything work with keras
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
sys.path.append("./../")
from tensorflow.python.keras.layers import Flatten
from tensorflow import keras
keras.backend.clear_session()
from pathlib import Path
import re
import mne
import matplotlib.pyplot as plt
import numpy as np
import yasa
import pandas as pd
from itertools import product
from SpectralImagesClassification.SpectralImagesUtils import gen_images

mne.set_log_level("WARNING")

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')

    return chName

#PO7 and PO8 removed
EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO3","PO4","OZ"]
renamedChannels = list(map(renameChannels,EEG_channels))
Power_coefficients = ['Theta','Alpha','Beta']
newColumnNames = [x+'-'+y for x,y in product(Power_coefficients,renamedChannels)]

def splitDataIntoEpochs(raw, frameDuration, overlap):
    # Split data into epochs
    w1 = frameDuration
    sf = 250
    totalPoints = raw.get_data().shape[1]
    nperE = sf * w1  # Number of samples per Epoch

    eTime = int(w1 / 2 * sf) + raw.first_samp
    events_array = []
    while eTime < raw.last_samp:
        events_array.append([eTime, 0, 1])
        eTime += sf * (w1 - overlap)

    events_array = np.array(events_array).astype(np.int)
    epochs = mne.Epochs(raw, events_array, tmin=-(w1 / 2), tmax=(w1 / 2))

    return epochs

def getBandPowerCoefficients(epoch_data):
    counter = 0
    dataDict = {}
    epoch_data.load_data()
    win_sec =0.95
    sf = 250

    for i in range(len(epoch_data)):
        data = epoch_data[i]
        data = data.get_data().squeeze() #Remove additional
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

def createImageDataset(path, imageSize,frameDuration, overlap,
                       image_format=True, lstm_format=False, lstm_sequence_length=0,
                       encoded_format=True, autoencoder=None,
                       augment_data=False, labels=None, fileNameFormat=1):

    if encoded_format:
        assert image_format, "encoded format requires eeg to be represented as images"
    if lstm_format:
        assert lstm_sequence_length > 0, "In lstm format sequence length needs to be set."

    #Load locations
    loc_2d = pd.read_csv('./channel_2d_location.csv', index_col='ch_name')
    loc_2d = loc_2d.loc[EEG_channels]
    loc_2d = loc_2d[['x','y']].values

    X = None
    y = None

    for idx, file in enumerate(path.rglob("*.edf")):

        if fileNameFormat == 1:
            uid = re.findall('.+(?=_S[0-9]{1}_T[0-9]{1}_)', file.name)[0]
            session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{1}_)', file.name)[0])
            trial = int(re.findall('(?<=_S[0-9]{1}_T)[0-9]{1}(?=_)', file.name)[0])
            task = re.findall('(?<=_S[0-9]{1}_T[0-9]{1}_).+(?=_pyprep\.edf)', file.name)[0]
        elif fileNameFormat == 2:
            #New calibration task
            uid = re.findall('.+(?=_S[0-9]{2}_T[0-9]{2}_)', file.name)[0]
            session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{2}_)', file.name)[0])
            trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]{2}(?=_)', file.name)[0])
            task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_raw\.edf)', file.name)[0]
        else:
            Exception("Wrong fileNameFormat!")

        if task == "Baseline":
            continue

        print("file information:", uid, session, trial, task)
        #Get Label
        if task == labels[0]:
            label = 0.0
        elif task == labels[1]:
            label = 1.0
        assert task == labels[0] or task == labels[1], "something went wrong with the labels. check {:}".format(file.name)
        # Read eeg file
        file = Path(file)
        raw = mne.io.read_raw_edf(file)
        mne.rename_channels(raw.info, renameChannels)
        #Remove bad channels
        raw = raw.pick(renamedChannels)
        # Filter data
        raw.load_data()
        raw.filter(0.5, 30)

        #Debug plot data
        # raw.plot()
        # plt.show()

        #Get epochs
        epochs = splitDataIntoEpochs(raw,frameDuration,overlap)
        bandpower = getBandPowerCoefficients(epochs)

        if image_format:
            images = gen_images(np.array(loc_2d), bandpower.values, imageSize, normalize=False, augment=augment_data)
            images = np.swapaxes(images, 1, 3)
            print(len(images), 'frames generated with label ', label)
            if encoded_format:
                images = autoencoder(images)
                images = Flatten()(images)
                images = images.numpy()
        else:
            #Keep the features in list format
            images = bandpower.values

        #Transform to a sequence of images that can be used with a lstm model.
        if lstm_format:
            images_sequences = []
            totalNumber = images.shape[0]
            idx2 = 0
            while idx2 + lstm_sequence_length < images.shape[0]:
                images_sequences.append(images[idx2:idx2+lstm_sequence_length])
                idx2 = idx2 + lstm_sequence_length
            images = np.array(images_sequences)

        # Append all the samples in a list
        if X is None:
            X = images
            y = np.ones(len(images)) * label
        else:
            X = np.concatenate((X, images), axis=0)
            y = np.concatenate((y, np.ones(len(images)) * label), axis=0)

    return X, np.array(y)

def showImages(X,y):
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0][0].set_title('Known Skill')
    axarr[0][0].imshow(X[6])
    axarr[1][0].imshow(X[20])

    axarr[0][1].set_title('Unknown Skill')
    axarr[0][1].imshow(X[38])
    axarr[1][1].imshow(X[40])
    plt.show()

if __name__ == "__main__":

    # dataPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiment1-Pilot\UI02\pyprep_edf")
    # dataPath = Path(r"./data")
    # X, y = createImageDataset(dataPath,32,1,0.0,image_format=False, lstm_format=False,
    #                           lstm_sequence_length=3, augment_data=True, labels = ["EasyAdd","HardMult"],
    #                           fileNameFormat=2)
    # print(X.shape)
    # showImages(X,y)

    img_size = 32
    frame_duration = 1
    overlap = 0.5

    dataPath = Path(r"./data")
    autoencoder = keras.models.load_model('autoEncoderWeights/autoencoder.h5')
    encoder = autoencoder.layers[1]
    X, y = createImageDataset(dataPath, imageSize=img_size,frameDuration=frame_duration,overlap=overlap,
                              image_format=True, augment_data=False, labels = ["EasyAdd","HardMult"],
                              encoded_format=True, autoencoder=encoder,lstm_format=True, lstm_sequence_length=5,
                              fileNameFormat=2)

    print(X.shape)


