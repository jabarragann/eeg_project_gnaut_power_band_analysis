import os
import traceback

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import re
from pathlib import Path
from random import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
from tensorflow.keras.models import load_model
import pickle
import matplotlib.cm as cm
import matplotlib.colors as col

Power_coefficients = ['Theta', 'Alpha', 'Beta']
EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO3","PO4","OZ"]

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

        self.dataBuffer = np.zeros((30000,30))+1.2
        self.sequenceForPrediction = np.zeros((1, self.lstm_sequence_length, 90))

        #Load prediction model and normalizer
        self.global_mean = self.normalization_dict['mean']
        self.global_std = self.normalization_dict['std']

        print("Deep model config")
        print("sf {:d} window length {:0.3f} overlap {:0.3f}"
              " lstm length {:d} Window size {:d} Chunk size {:d}".format(  self.sf,
                                                                            self.window_length,
                                                                            self.overlap,
                                                                            self.lstm_sequence_length,
                                                                            self.window_size,
                                                                            self.chunk_size))
    def make_prediction(self, chunk):

        ###################
        ## Get new samples#
        ###################
        chunk = np.array(chunk) #Should be of size (125,30)
        # Roll old data to make space for the new chunk
        self.dataBuffer = np.roll(self.dataBuffer, -self.chunk_size, axis=0)  # Buffer Shape (Time*SF, channels)
        # Add chunk in the last 125 rows of the data buffer. 250 samples ==> 1 second.
        self.dataBuffer[-self.chunk_size:, :] = chunk[:, :] # Add chunk in the last 125 rows. 250 samples ==> 1 second.

        ######################
        ##Calculate Features##
        ######################
        # Check that data is in the correct range
        assert all([0.4 < abs(self.dataBuffer[-2*self.chunk_size:, 2].min()) < 800,
                    1 < abs(self.dataBuffer[-2*self.chunk_size:, 7].max()) < 800,
                    0.4 < abs(self.dataBuffer[-2*self.chunk_size:, 15].min()) < 800]), \
            "Check the units of the data that is about to be process. " \
            "Data should be given as uv to the get bandpower coefficients function "

        # Get bandPower coefficients
        win_sec = 0.95
        bandpower = yasa.bandpower(self.dataBuffer[-2*self.chunk_size:, :].transpose(),
                                   sf=self.sf, ch_names=EEG_channels, win_sec=win_sec,
                                   bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 40, 'Beta')])
        bandpower = bandpower[Power_coefficients].transpose()
        bandpower = bandpower.values.reshape(1, -1)

        # Add feature to vector to LSTM sequence and normalize sequence for prediction.
        self.sequenceForPrediction = np.roll(self.sequenceForPrediction, -1, axis=1) #Sequence shape (1, timesteps, #features)
        self.sequenceForPrediction[0, -1, :] = bandpower  # Set new data point in last row

        #normalize sequence
        normalSequence = (self.sequenceForPrediction - self.global_mean)/self.global_std

        prediction = self.predictionModel.predict(normalSequence)
        prediction = prediction[0, 1]

        predicted_label = 1 if prediction > 0.5 else 0

        print("prediction scores:", prediction, "label: ", predicted_label)

        return prediction, self.sequenceForPrediction, self.dataBuffer[-2*self.chunk_size:, :]


def main():
    print(random())
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\EyesOpen_closeDataset\edfjuan-03-07-21-validation")

    # Open EEG and video ts files, and video
    eeg_file_name = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_.+_raw.txt", f.name))>0 ][0]
    ts_file = pd.read_csv(srcPath / (eeg_file_name.with_suffix('').name + "_video_right_ts.txt"))
    task = eeg_file_name.with_suffix("").name

    print("loading eeg from {:}".format(eeg_file_name.name))
    eeg_file = pd.read_csv(eeg_file_name)
    # create Graph generator
    deep_model = simple_lstm_predictor("simple_lstm_seq5_eyes")
    # deep_model = simple_lstm_predictor("simple_lstm_seq25_Juan-needle-vs-needleBlood-last-try")


    prev_time = 0
    prediction_df = pd.DataFrame(columns=['LSL_TIME','MODEL_PREDICTION'])
    features_array = []
    raw_array = []
    lsl_timestamps = []
    total_eeg_pt = 0
    eeg_data = np.zeros((30, 500))
    initial_ts = ts_file.loc[0, 'ecm_ts']

    try:
        for idx in range(ts_file.shape[0]):
            # Get EEG data from file
            ts = ts_file.loc[idx, 'ecm_ts']
            data = eeg_file.loc[(eeg_file["COMPUTER_TIME"] > prev_time) & (eeg_file["COMPUTER_TIME"] < ts)]
            lsl_time = data["LSL_TIME"].values[-1]

            count_of_eeg = data.shape[0]
            total_eeg_pt += count_of_eeg
            prev_time = ts

            #Update eeg data buffer
            new_data = data[EEG_channels].values
            shift = new_data.shape[0]
            eeg_data = np.roll(eeg_data, -shift, axis=1)
            eeg_data[:, -shift:] = new_data.transpose()

            # Create graph for predictions
            if total_eeg_pt > 125:
                print("Make prediction")
                print("time: {:f} total eeg points: {:d}".format(ts - initial_ts, total_eeg_pt))
                new_chunk = eeg_data[:, -total_eeg_pt:][:, 0:125]
                prediction, features, raw = deep_model.make_prediction(new_chunk.transpose())
                total_eeg_pt -= 125
                # Debug
                prediction_df.loc[idx] = lsl_time,prediction
                lsl_timestamps.append(lsl_time)
                features_array.append(features)
                raw_array.append(raw)

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        prediction_df.to_csv('CheckPredictionPlot/{:}_model_predictions.csv'.format(task))
        print("Finished")


if __name__ == "__main__":
    main()