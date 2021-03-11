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
import random

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

        #Check that data is in the correct range
        assert all([1 < abs(data[4, :].min()) < 800,
                    1 < abs(data[7, :].max()) < 800,
                    1 < abs(data[15, :].min()) < 800]), \
                    "Check the units of the data that is about to be process. " \
                    "Data should be given as uv to the get bandpower coefficients function "

        # Calculate bandpower # Data should always be process in the uv
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

def createSequencesForLstm(images, lstm_sequence_length, overlapping_sequences = True, stride = 1 ):
    """
        Transform list of images into sequences for a LSTM model.
    :param overlapping_sequences:
    :param lstm_sequence_length:
    :param images:
    :return:
    """
    images_sequences = []
    idx2 = 0
    while idx2 + lstm_sequence_length < images.shape[0]:
        images_sequences.append(images[idx2:idx2 + lstm_sequence_length])
        if overlapping_sequences:
            idx2 = idx2 + stride
        else:
            idx2 = idx2 + lstm_sequence_length

    images = np.array(images_sequences)

    return images

def createImageDataset(path, frame_duration, overlap,
                       lstm_format=False, lstm_sequence_length=0, lstm_stride=1, sequence_overlap = False,
                       image_format=False, image_size = None, encoded_format=False, autoencoder=None,
                       augment_data=False, labels=None, file_name_format=1, read_from='path'):
    """
    :param sequence_overlap:
    :param path: If read_from is set to 'path' then this variable should be a path to a directory containing all the files.
                Else if read_from is set to 'files' then this variable should be a list of of files.
    :param image_size:
    :param frame_duration:
    :param overlap:
    :param image_format:
    :param lstm_format:
    :param lstm_sequence_length:
    :param lstm_stride:
    :param encoded_format:
    :param autoencoder:
    :param augment_data:
    :param labels:
    :param file_name_format:
    :param read_from: either 'path' or 'files' depending where is the data going to be load from.
    :return:
    """

    assert read_from == 'path' or read_from == 'files', "'read_from' has an incorrect values"
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

    generator = path.rglob("*.edf") if read_from == 'path' else path

    for idx, file in enumerate(generator):
        if file_name_format == 1:
            uid = re.findall('.+(?=_S[0-9]{1}_T[0-9]{1}_)', file.name)[0]
            session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{1}_)', file.name)[0])
            trial = int(re.findall('(?<=_S[0-9]{1}_T)[0-9]{1}(?=_)', file.name)[0])
            task = re.findall('(?<=_S[0-9]{1}_T[0-9]{1}_).+(?=_pyprep\.edf)', file.name)[0]
        elif file_name_format == 2:
            #New calibration task
            uid = re.findall('.+(?=_S[0-9]{2}_T[0-9]{2}_)', file.name)[0]
            session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{2}_)', file.name)[0])
            trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]{2}(?=_)', file.name)[0])
            task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_raw\.edf)', file.name)[0]
        else:
            Exception("Wrong file_name_format!")

        if task not in labels:
            print("Skipping {:}".format(str(file.name)))
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
        epochs = splitDataIntoEpochs(raw, frame_duration, overlap)
        bandpower = getBandPowerCoefficients(epochs)

        if image_format:
            images = gen_images(np.array(loc_2d), bandpower.values, image_size, normalize=False, augment=augment_data)
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
            images = createSequencesForLstm(images,lstm_sequence_length,overlapping_sequences=sequence_overlap, stride=lstm_stride)

        # Append all the samples in a list
        if X is None:
            X = images
            y = np.ones(len(images)) * label
        else:
            X = np.concatenate((X, images), axis=0)
            y = np.concatenate((y, np.ones(len(images)) * label), axis=0)

    return X, np.array(y)

def loadSingleTxtFile(filePathLib, imageSize,frameDuration, overlap,
                   image_format=False, lstm_format=False, lstm_sequence_length=0,
                   encoded_format=False, autoencoder=None,
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

    uid = re.findall('.+(?=_S[0-9]{2}_T[0-9]{2}_)', filePathLib.name)[0]
    session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{2}_)', filePathLib.name)[0])
    trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]{2}(?=_)', filePathLib.name)[0])
    task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_raw\.txt)', filePathLib.name)[0]
    print("file information:", uid, session, trial, task)

    #Get Label
    label = None
    if task == labels[0]:
        label = 0.0
    elif task == labels[1]:
        label = 1.0
    assert task == labels[0] or task == labels[1], "something went wrong with the labels. check {:}".format(filePathLib.name)
    # Read eeg file
    eeg_file = pd.read_csv(filePathLib)
    data = eeg_file[EEG_channels].values.transpose()
    data = data
    ch_names = EEG_channels
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    mne.rename_channels(raw.info, renameChannels)
    raw = raw.pick(renamedChannels)  #Remove bad channels
    # Filter data
    raw.load_data()
    raw.filter(0.5, 30)

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
        images = createSequencesForLstm(images,lstm_sequence_length,overlapping_sequences=False)

    # Append all the samples in a list
    if X is None:
        X = images
        y = np.ones(len(images)) * label
    else:
        X = np.concatenate((X, images), axis=0)
        y = np.concatenate((y, np.ones(len(images)) * label), axis=0)

    return X, np.array(y)

def buildTrainAndVal(path, split_rate,frame_duration, overlap,
                     image_size=None,image_format=False,encoded_format=False, autoencoder=None,
                     lstm_format=False, lstm_sequence_length=0, lstm_stride=1, sequence_overlap=False,
                     augment_data=False, labels=None, file_name_format=1):
    """
    Create train and testing sets from different eeg data files. Some files are exclusively use for training will others are used
    for testing.

    :param sequence_overlap:
    :param image_size:
    :param frame_duration:
    :param lstm_format:
    :param lstm_stride:
    :param lstm_sequence_length:
    :param file_name_format:
    :param autoencoder:
    :param augment_data:
    :param encoded_format:
    :param image_format:
    :param overlap:
    :param frame_duration:
    :param labels: labels of the files.
    :param path: path to search files
    :param split_rate:
    :return: x_train, x_test, y_train, y_test
    """
    label1Files = np.array(list(path.rglob("*{:}*".format(labels[0]))))
    label2Files = np.array(list(path.rglob("*{:}*".format(labels[1]))))

    train_size = int(len(label1Files)*(1-split_rate))


    idx = list(range(len(label1Files)))

    train_idx = random.sample(idx,k=train_size)
    [idx.remove(i) for i in train_idx]
    val_idx = idx
    trainFiles = np.concatenate((label1Files[train_idx], label2Files[train_idx]))
    valFiles = np.concatenate((label1Files[val_idx], label2Files[val_idx]))

    print("Loading train")
    x_train,y_train = createImageDataset(trainFiles, image_size=image_size, frame_duration=frame_duration,
                                         overlap=overlap, image_format=image_format, lstm_format=lstm_format,
                                         lstm_sequence_length=lstm_sequence_length, lstm_stride=lstm_stride, sequence_overlap=sequence_overlap,
                                         encoded_format=encoded_format, autoencoder=autoencoder, augment_data=augment_data,
                                         labels=labels, file_name_format=file_name_format, read_from='files')
    print("Loading test")
    x_test,y_test = createImageDataset(valFiles, image_size=image_size, frame_duration=frame_duration,
                                       overlap=overlap, image_format=image_format, lstm_format=lstm_format,
                                       lstm_sequence_length=lstm_sequence_length, lstm_stride=lstm_stride, sequence_overlap=sequence_overlap,
                                       encoded_format=encoded_format, autoencoder=autoencoder, augment_data=augment_data,
                                       labels=labels, file_name_format=file_name_format, read_from='files')

    return x_train, x_test, y_train, y_test

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


def main():
    # dataPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiment1-Pilot\UI02\pyprep_edf")
    dataPath = Path(r"./data")
    image_size = 32
    frame_length = 1
    sequence_length = 6
    overlap = 0.5
    num_classes = 2

    #
    # x_train, x_test, y_train, y_test = buildTrainAndVal(dataPath,split_rate=0.20, frame_duration= frame_length,overlap=overlap,
    #                                                     lstm_format=True,lstm_sequence_length=sequence_length,lstm_stride=2, sequence_overlap=True,
    #                                                     labels = ["EasyAdd", "HardMult"], file_name_format=2)
    # print("Train", x_train.shape, y_train.shape)
    # print("Test", x_test.shape, y_test.shape)

    # image_size = 32
    # frame_length = 1
    # sequence_length = 15
    # overlap = 0.5
    # num_classes = 2
    #
    # dataPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-SurgicalTasks\edf\Jing")
    # x_train, x_test, y_train, y_test = buildTrainAndVal(dataPath, split_rate=0.20, frame_duration=frame_length,
    #                                                     overlap=overlap,
    #                                                     lstm_format=True, lstm_sequence_length=sequence_length,
    #                                                     lstm_stride=2, sequence_overlap=True,
    #                                                     labels=["PegTransfer", "KnotTying"], file_name_format=2)
    #
    # X, y = createImageDataset(dataPath, image_size=image_size, frame_duration=frame_length,
    #                           overlap=overlap,
    #                           image_format=False, encoded_format=False, augment_data=False,
    #                           labels=["EasyAdd", "HardMult"],
    #                           lstm_format=True, lstm_sequence_length=sequence_length,
    #                           file_name_format=2)
    # print(X.shape)
    # showImages(X,y)

    # img_size = 32
    # frame_duration = 1
    # overlap = 0.5
    #
    # dataPath = Path(r"./data")
    # autoencoder = keras.models.load_model('autoEncoderWeights/autoencoder.h5')
    # encoder = autoencoder.layers[1]
    # X, y = createImageDataset(dataPath, image_size=img_size,frame_duration=frame_duration,overlap=overlap,
    #                           image_format=True, augment_data=False, labels = ["EasyAdd","HardMult"],
    #                           encoded_format=True, autoencoder=encoder,lstm_format=True, lstm_sequence_length=5,
    #                           file_name_format=2)
    #
    # print(X.shape)

    # srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data")
    # srcPath = srcPath / r"TestsWithVideo\Eyes-open-close-test\T01"
    # srcPath = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name)) > 0][0]
    # print("loading eeg from {:}".format(srcPath.name))
    # X, y = loadSingleTxtFile(srcPath, image_size=32, frame_duration=1,
    #                          overlap=0.5,
    #                          image_format=False, encoded_format=False, augment_data=False,
    #                          labels=["Eyes-open-close", ""],
    #                          lstm_format=True, lstm_sequence_length=1,
    #                          file_name_format=2)

    image_size = 32
    frame_length = 1
    sequence_length = 25  # 20
    overlap = 0.5
    num_classes = 2

    dataPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\edf\Jing")

    x_train, x_test, y_train, y_test = buildTrainAndVal(dataPath, split_rate=0.20, frame_duration=frame_length,
                                                        overlap=overlap,
                                                        lstm_format=True, lstm_sequence_length=sequence_length,
                                                        lstm_stride=2, sequence_overlap=True,
                                                        labels=["NeedlePassing", "BloodNeedlePassing"],
                                                        file_name_format=2)


def test():
    print("File loaded")

if __name__ == "__main__":
    main()
    # test()

