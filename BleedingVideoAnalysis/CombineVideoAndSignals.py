import os
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

class simple_conv_predictor:
    def __init__(self):
        self.sf = 250
        self.window_length = 1.0
        self.overlap = 0.5
        self.img_size = 64

        self.window_size = int(self.sf * self.window_length)
        self.chunk_size = int(self.sf * self.window_length - self.sf * self.overlap)

        self.dataBuffer = np.zeros((1500, 30))
        self.sequenceForPrediction = np.zeros((1, self.img_size,self.img_size, 3))

        # Load prediction model and normalizer
        self.predictionModel = load_model('./deep_model/conv_eyes-open-close_64px.h5')
        self.normalization_dict = pickle.load(open('./deep_model/conv_eyes-open-close_64px_normalizer.pickle', 'rb'))
        self.global_mean = self.normalization_dict['mean']
        self.global_std = self.normalization_dict['std']
    def make_prediction(self,chunk):
        ###################
        ## Get new samples#
        ###################
        chunk = np.array(chunk)  # Should be of size (125,30)
        # Roll old data to make space for the new chunk
        dataBuffer = np.roll(self.dataBuffer, -self.chunk_size, axis=0)  # Buffer Shape (Time*SF, channels)
        # Add chunk in the last 125 rows of the data buffer. 250 samples ==> 1 second.
        dataBuffer[-self.chunk_size:, :] = chunk[:, :]  # Add chunk in the last 125 rows. 250 samples ==> 1 second.

        ######################
        ##Calculate Features##
        ######################
        # Get bandPower coefficients
        spectral_img = self.conver_eeg_to_img(dataBuffer[-2 * self.chunk_size:, :].transpose())

        # Add feature to vector to LSTM sequence and normalize sequence for prediction.
        self.sequenceForPrediction = spectral_img

        # normalize sequence
        normalSequence = (self.sequenceForPrediction - self.global_mean) / self.global_std

        prediction = self.predictionModel.predict(normalSequence)
        prediction = prediction[0, 1]

        predicted_label = 1 if prediction > 0.5 else 0

        print("prediction scores:", prediction, "label: ", predicted_label)

        return prediction

    def convert_eeg_to_img(self, data):
        win_sec = 0.95
        bd = yasa.bandpower(data, sf=self.sf, ch_names=EEG_channels, win_sec=win_sec,
                            bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 40, 'Beta')])
        # Reshape coefficients into a single row vector with the format
        # [Fp1Theta,Fp2Theta,AF3Theta,.....,Fp1Alpha,Fp2Alpha,AF3Alpha,.....,Fp1Beta,Fp2Beta,AF3Beta,.....,]
        bandpower = bd[Power_coefficients].transpose()
        bandpower = bandpower.values.reshape(1, -1)

        #Continue image ...

        return np.squeeze(bandpower)

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

        self.dataBuffer = np.zeros((30000,30))
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
        # Get bandPower coefficients
        win_sec = 0.95
        bandpower = yasa.bandpower(self.dataBuffer[-2*self.chunk_size:, :].transpose() * 1e6,
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


class GraphGenerator:
    def __init__(self):
        #EEG data
        self.eeg_data = np.zeros((30,2500))
        self.threshold = 0.30

        # Raw data plotting
        self.fig, self.ax = plt.subplots(2,2, sharex='col',sharey='row')
        self.fig.tight_layout()
        self.x1 = np.linspace(2.0, 14.0,2500)

        self.channels_to_display = ["FP1","T7","CP5","OZ"]
        self.channels_to_display_idx = [np.where(np.array(EEG_channels) == c)[0][0] for c in self.channels_to_display]
        self.lines = self.create_lines(["red","blue","green","orange"])
        self.format_graphs(self.channels_to_display)

        #Prediction graph
        self.prediction_fig, self.prediction_ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
        self.prediction_ax[0].set_ylim(-0.05, 1.3)
        self.prediction_ax[0].set_xlim(0.0, 65)

        #format for barchart
        self.cmapHigh = cm.ScalarMappable(col.Normalize(-0, 1), 'RdBu_r')
        self.cmapLow = cm.ScalarMappable(col.Normalize(-0, 1), 'RdBu')

        self.predictions = np.zeros(80)
        self.labels = np.zeros_like(self.predictions)
        self.x1 = np.linspace(0.0, 65.0, 80)

        self.prediction_line,self.label_line,self.prediction_bar = self.create_prediction_lines()
        self.format_prediction_plots()

    def create_prediction_lines(self):
        pred_line, = self.prediction_ax[0].plot(self.x1, self.predictions, 'ko-', linewidth=0.5, label='Prediction')
        label_line, = self.prediction_ax[0].plot(self.x1, self.predictions, 'gs', linestyle='None', markersize=3.0, label='label')
        threshold_line, = self.prediction_ax[0].plot(self.x1, np.ones_like(self.x1)*self.threshold, 'b--', label='threshold')
        self.prediction_ax[0].legend(loc='upper left', fontsize='x-small')

        colorForBars = np.array([self.cmapLow.to_rgba(1), self.cmapHigh.to_rgba(0)])
        pred_bar = self.prediction_ax[1].bar([1, 2], [0.0,0.7], width=1.0, color=colorForBars)

        return [pred_line,label_line,pred_bar]
    def format_prediction_plots(self):
        #Prediction plot format
        # self.prediction_ax[0].set_title("LSTM predictions")
        self.prediction_ax[0].spines["top"].set_visible(False)
        self.prediction_ax[0].spines["right"].set_visible(False)
        self.prediction_ax[0].spines["left"].set_visible(False)
        self.prediction_ax[0].set_xticks([0,10,20,30,40,50,60])
        self.prediction_ax[0].set_xticklabels([])
        self.prediction_ax[0].set_yticks([])
        self.prediction_ax[0].set_yticklabels([])
        #bar Chart format
        self.prediction_ax[1].set_title("Average")
        self.prediction_ax[1].set_xlim((0,3))
        self.prediction_ax[1].set_xticks([]) #[0, 1, 2, 3]
        self.prediction_ax[1].set_xticklabels([]) #['', 'L', 'H', '']
        self.prediction_ax[1].set_yticks([])
        self.prediction_ax[1].set_yticklabels([])
        self.prediction_ax[1].spines["top"].set_visible(False)
        self.prediction_ax[1].spines["right"].set_visible(False)
        self.prediction_ax[1].spines["left"].set_visible(False)
        self.prediction_ax[1].spines["bottom"].set_visible(True)

    def create_lines(self, colors):
        lines = []
        for a,c in zip(self.ax.reshape(-1),colors):
            line, = a.plot(self.x1, self.eeg_data[0, :],color=c, linewidth=0.5)
            lines.append(line)
        return lines

    def format_graphs(self, channels):
        for a, ch in zip(self.ax.reshape(-1), channels):
            a.set_title(ch)
            a.set_ylim((-120, 120))
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            a.set_xticks([])
            a.set_xticks([],minor=True)
            a.set_yticks([])
            a.set_yticks([],minor=True)

    def update_eeg_graph(self):
        for l, idx in zip(self.lines,self.channels_to_display_idx):
            l.set_ydata(self.eeg_data[idx,:].squeeze())

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = self.create_img_from_graph(self.fig)
        return img
    def update_prediction_graph(self):
        self.prediction_line.set_ydata(self.predictions.squeeze())
        self.label_line.set_ydata(self.labels.squeeze())
        self.prediction_fig.canvas.draw()
        #self._update_barchart(self.predictions.squeeze()[-1])
        img = self.create_img_from_graph(self.prediction_fig)
        return img
    def _update_barchart(self,value):
        self.prediction_bar[0].set_height(1 - value)
        self.prediction_bar[1].set_height(value)
        self.prediction_bar[0].set_color(self.cmapLow.to_rgba(1 - value))
        self.prediction_bar[1].set_color(self.cmapHigh.to_rgba(value))

    def update_eeg_data(self, new_data):
        shift = new_data.shape[0]
        self.eeg_data = np.roll(self.eeg_data, -shift, axis=1)
        self.eeg_data[:, -shift:] = new_data.transpose()
    def update_predictions_data(self, new_data):
        self.predictions = np.roll(self.predictions, -1)
        self.predictions[-1] = new_data
        l = 1.0 if new_data > self.threshold else 0.0
        self.labels = np.roll(self.labels, -1)
        self.labels[-1] = l

    @staticmethod
    def create_img_from_graph(fig):
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
        img = cv2.resize(img, (640, 480))
        return img.astype(np.uint8)


def main():
    print(random())
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data")
    srcPath = srcPath / r"TestsWithVideo\Eyes-open-close-test\T01"

    # Output video
    resultPath = Path("./video") / "out.avi"
    frame_width = 640 * 3
    frame_height = 480
    out = cv2.VideoWriter(str(resultPath), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    # Open EEG and video ts files, and video
    eeg_file = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name))>0 ][0]
    print("loading eeg from {:}".format(eeg_file.name))
    eeg_file = pd.read_csv(eeg_file)
    ts_file = pd.read_csv(srcPath / "video_right_color_ts_trimmed.txt")

    cap = cv2.VideoCapture(str(srcPath / "video_right_color_trimmed.avi"))

    # create Graph generator
    deep_model = simple_lstm_predictor("simple_lstm_seq5_eyes")
    graph_generator = GraphGenerator()
    prediction_graph = graph_generator.update_prediction_graph()

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    count = 0
    prev_time = 0
    prediction_array = []
    features_array = []
    raw_array = []
    initial_ts = ts_file.loc[count, 'ecm_ts']
    total_eeg_pt = 0
    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame

        if ret:
            # Get EEG data from file
            ts = ts_file.loc[count, 'ecm_ts']
            data = eeg_file.loc[(eeg_file["COMPUTER_TIME"] > prev_time) & (eeg_file["COMPUTER_TIME"] < ts)]
            count_of_eeg = data.shape[0]
            total_eeg_pt += count_of_eeg
            prev_time = ts

            # Create plots
            graph_generator.update_eeg_data(data[EEG_channels].values*1e6)
            # graph_from_signals = (np.ones((480,640,3))*255).astype(np.uint8)
            graph_from_signals = graph_generator.update_eeg_graph()  # np.ones((480,640,3))*255

            # Create graph for predictions
            if total_eeg_pt > 125:
                print("Make prediction")
                new_chunk = graph_generator.eeg_data[:, -total_eeg_pt:][:, 0:125]
                prediction, features, raw = deep_model.make_prediction(new_chunk.transpose())
                graph_generator.update_predictions_data(prediction)
                prediction_graph = graph_generator.update_prediction_graph()
                total_eeg_pt -= 125

                #Debug
                prediction_array.append(prediction)
                features_array.append(features)
                raw_array.append(raw)
            count += 1

            print("time: {:f} total eeg points: {:d}".format(ts - initial_ts, total_eeg_pt))
            final_frame = np.hstack(( graph_from_signals,frame, prediction_graph,))
            out.write(final_frame)
            cv2.imshow('Frame', final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    pickle.dump(prediction_array, open('CheckPredictionPlot/array_of_pred.pickle', 'wb'))
    pickle.dump(features_array, open('CheckPredictionPlot/array_of_features.pickle', 'wb'))
    pickle.dump(raw_array, open('CheckPredictionPlot/array_of_raw.pickle', 'wb'))
    pickle.dump(deep_model.dataBuffer, open('CheckPredictionPlot/data_buffer.pickle', 'wb'))
if __name__ == "__main__":
    main()