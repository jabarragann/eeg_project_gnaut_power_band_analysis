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

class simple_lstm_predictor:
    def __init__(self, lstm_model_name):
        #Load model and configuration
        self.predictionModel = load_model('./deep_model/{:}.h5'.format(lstm_model_name))
        self.normalization_dict = pickle.load(open('./deep_model/{:}_normalizer.pickle'.format(lstm_model_name), 'rb'))
        self.configuration_dict = pickle.load(open('./deep_model/{:}_config.pickle'.format(lstm_model_name), 'rb'))

        self.sf =250
        self.window_length = self.configuration_dict['frame_length']
        self.overlap = self.configuration_dict['overlap']
        self.lstm_sequence_length = self.configuration_dict['sequence_length']

        self.window_size = int(self.sf*self.window_length)
        self.chunk_size  = int(self.sf*self.window_length - self.sf*self.overlap)


        self.dataBuffer = np.zeros((1500,30))
        self.sequenceForPrediction = np.zeros((1, self.lstm_sequence_length, 90))

        #Load prediction model and normalizer
        self.global_mean = self.normalization_dict['mean']
        self.global_std = self.normalization_dict['std']
    def make_prediction_simple(self,chunk):

        ######################
        ##Calculate Features##
        ######################
        # # Get bandPower coefficients
        chunk *= 1e6
        win_sec = 0.95
        bandpower = yasa.bandpower(chunk.transpose(),
                                   sf=self.sf, ch_names=EEG_channels, win_sec=win_sec,
                                   bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 40, 'Beta')])
        bandpower = bandpower[Power_coefficients].transpose()
        bandpower = bandpower.values.reshape(1, -1)

        #with func
        # bandpower = getBandPowerCoefficients(chunk)
        # Reshape coefficients into a single row vector
        # bandpower = bandpower[Power_coefficients].values.reshape(1, -1)

        # Add feature to vector to LSTM sequence and normalize sequence for prediction.
        self.sequenceForPrediction = np.roll(self.sequenceForPrediction, -1,
                                             axis=1)  # Sequence shape (1, timesteps, #features)
        self.sequenceForPrediction[0, -1, :] = bandpower  # Set new data point in last row

        # normalize sequence
        normalSequence = (self.sequenceForPrediction - self.global_mean) / self.global_std

        prediction = self.predictionModel.predict(normalSequence)
        prediction = prediction[0, 1]

        predicted_label = 1 if prediction > 0.5 else 0

        print("prediction scores:", prediction, "label: ", predicted_label)

        return prediction

    def make_prediction(self, chunk):

        ###################
        ## Get new samples#
        ###################
        chunk = np.array(chunk) #Should be of size (125,30)
        # Roll old data to make space for the new chunk
        dataBuffer = np.roll(self.dataBuffer, -self.chunk_size, axis=0)  # Buffer Shape (Time*SF, channels)
        # Add chunk in the last 125 rows of the data buffer. 250 samples ==> 1 second.
        dataBuffer[-self.chunk_size:, :] = chunk[:, :] # Add chunk in the last 125 rows. 250 samples ==> 1 second.

        ######################
        ##Calculate Features##
        ######################
        # Get bandPower coefficients
        win_sec = 0.95
        bandpower = yasa.bandpower(dataBuffer[-2*self.chunk_size:, :].transpose(),
                                   sf=self.sf, ch_names=EEG_channels, win_sec=win_sec,
                                   bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 40, 'Beta')])

        # Reshape coefficients into a single row vector
        bandpower = bandpower[Power_coefficients].values.reshape(1, -1)

        # Add feature to vector to LSTM sequence and normalize sequence for prediction.
        self.sequenceForPrediction = np.roll(self.sequenceForPrediction, -1, axis=1) #Sequence shape (1, timesteps, #features)
        self.sequenceForPrediction[0, -1, :] = bandpower  # Set new data point in last row

        #normalize sequence
        normalSequence = (self.sequenceForPrediction - self.global_mean)/self.global_std

        prediction = self.predictionModel.predict(normalSequence)
        prediction = prediction[0, 1]

        predicted_label = 1 if prediction > 0.5 else 0

        print("prediction scores:", prediction, "label: ", predicted_label)

        return prediction, self.sequenceForPrediction, dataBuffer[-2*self.chunk_size:, :]

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

def loadSingleTxtFile(filePathLib,frameDuration, overlap, lstm_sequence_length=0):

    X = None
    y = None

    # Read eeg file
    eeg_file = pd.read_csv(filePathLib)
    data = eeg_file[EEG_channels].values.transpose()
    data = data
    ch_names = EEG_channels
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    mne.rename_channels(raw.info, renameChannels)
    renamed = list(map(renameChannels,EEG_channels))
    raw = raw.pick(renamed)  #Remove bad channels
    # Filter data
    raw.load_data()
    raw.filter(0.5, 30)

    #Get epochs
    epochs = splitDataIntoEpochs(raw,frameDuration,overlap)
    bandpower = getBandPowerCoefficients(epochs.get_data())


    images = bandpower.values

    #LSTM format
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
    else:
        X = np.concatenate((X, images), axis=0)

    return X, epochs.get_data()


def generate_new_pre(raw, model):
    new = []
    for d in range(raw.shape[0]):
        new.append(model.make_prediction_simple(raw[d]))
    return new

def predict_in_a_different(lstm_model_name, raw):
    stm_model_name = "simple_lstm_seq1_eyes"
    predictionModel = load_model('./deep_model/{:}.h5'.format(lstm_model_name))
    normalization_dict = pickle.load(open('./deep_model/{:}_normalizer.pickle'.format(lstm_model_name), 'rb'))
    global_mean = normalization_dict['mean']
    global_std = normalization_dict['std']
    configuration_dict = pickle.load(open('./deep_model/{:}_config.pickle'.format(lstm_model_name), 'rb'))

    # Test model
    X_normalized = (raw - global_mean) / global_std

    predictions = predictionModel.predict(X_normalized)
    predictions = predictions[:, 1]
    return predictions

def main():
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data")
    srcPath = srcPath / r"TestsWithVideo\Eyes-open-close-test\T01"
    srcPath = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name)) > 0][0]
    print("loading eeg from {:}".format(srcPath.name))
    X, epochs = loadSingleTxtFile(srcPath, frameDuration=1, overlap=0.5, lstm_sequence_length=1, )

    pred_array = pickle.load(open('../CheckPredictionPlot/array_of_pred.pickle', 'rb'))
    pred_array = np.array(pred_array)
    features_array = pickle.load(open('../CheckPredictionPlot/array_of_features.pickle', 'rb'))
    raw_array = pickle.load(open('../CheckPredictionPlot/array_of_raw.pickle', 'rb'))
    raw_array = np.array(raw_array)

    model = simple_lstm_predictor("simple_lstm_seq1_eyes")
    new_pred = np.array(generate_new_pre(epochs.transpose((0, 2, 1)),model))

    # Alternative pred
    # alternative_pred = predict_in_a_different("simple_lstm_seq1_eyes",X)
    alternative_pred_1 = predict_in_a_different("simple_lstm_seq1_eyes", np.concatenate(features_array))
    alternative_pred_2 = predict_in_a_different("simple_lstm_seq1_eyes", X)

    fig, axes = plt.subplots(3, 1)
    colors = np.where(pred_array > 0.35, 'y', 'k')
    x = list(range(pred_array.shape[0]))
    axes[0].scatter(x, pred_array, c=colors)
    colors = np.where(new_pred > 0.35, 'b', 'r')
    axes[0].scatter(x[1:], new_pred, c=colors)

    # Alternative predictions
    x = list(range(alternative_pred_1.shape[0]))
    colors = np.where(alternative_pred_1 > 0.35, 'y', 'k')
    axes[1].scatter(x, alternative_pred_1, c=colors)

    x = list(range(alternative_pred_2.shape[0]))
    colors = np.where(alternative_pred_2 > 0.35, 'y', 'k')
    axes[2].scatter(x, alternative_pred_2, c=colors)
    plt.show()

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

