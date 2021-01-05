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
import traceback
from BleedingTestV2.Utils import clean_axes, create_cbar

Power_coefficients = ['Delta','Theta', 'Alpha', 'Beta']
EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO3","PO4","OZ"]


class feature_factory:
    def __init__(self, window_length, window_overlap):

        self.sf =250
        self.window_length = window_length
        self.overlap = window_overlap
        self.window_size = int(self.sf*self.window_length)
        self.incoming_chunk_size  = int(self.sf*self.window_length - self.sf*self.overlap)

        self.dataBuffer = np.zeros((30000,30))+1.2
        self.mean_rolling_fz = np.zeros((4,15))
        self.mean_rolling_t8 = np.zeros((4,15))
        self.mean_rolling_t7 = np.zeros((4,15))

    def make_prediction(self, chunk):

        ###################
        ## Get new samples#
        ###################
        chunk = np.array(chunk) #Should be of size (new_incoming_chunk_size,30)
        # Roll old data to make space for the new chunk
        self.dataBuffer = np.roll(self.dataBuffer, -self.incoming_chunk_size, axis=0)  # Buffer Shape (Time*SF, channels)
        # Add chunk in the last 125 rows of the data buffer. 250 samples ==> 1 second.
        self.dataBuffer[-self.incoming_chunk_size:, :] = chunk[:, :] # Add chunk in the last 125 rows. 250 samples ==> 1 second.

        ######################
        ##Calculate Features##
        ######################
        # Check that data is in the correct range
        assert all([0.4 < abs(self.dataBuffer[-2*self.incoming_chunk_size:, 2].min()) < 800,
                    1 < abs(self.dataBuffer[-2*self.incoming_chunk_size:, 7].max()) < 800,
                    0.4 < abs(self.dataBuffer[-2*self.incoming_chunk_size:, 15].min()) < 800]), \
            "Check the units of the data that is about to be process. " \
            "Data should be given as uv to the get bandpower coefficients function "

        # Get bandPower coefficients
        win_sec = 0.95 * self.window_length
        bandpower = yasa.bandpower(self.dataBuffer[-2*self.incoming_chunk_size:, :].transpose(),
                                   sf=self.sf, ch_names=EEG_channels, win_sec=win_sec,
                                   bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                          (12, 30, 'Beta'), (30, 50, 'Gamma')])


        bandpower = bandpower[Power_coefficients].transpose()

        self.mean_rolling_fz = np.roll(self.mean_rolling_fz, -1, axis=1)
        self.mean_rolling_t8 = np.roll(self.mean_rolling_t8, -1, axis=1)
        self.mean_rolling_t7 = np.roll(self.mean_rolling_t7, -1, axis=1)

        self.mean_rolling_fz[:,-1] = bandpower["FZ"].values
        self.mean_rolling_t8[:,-1] = bandpower["T8"].values
        self.mean_rolling_t7[:,-1] = bandpower["T7"].values

        return [self.mean_rolling_fz.mean(axis=1)[0], self.mean_rolling_t8.mean(axis=1)[0],
                self.mean_rolling_t7.mean(axis=1)[0]]


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

        #topo graph
        self.topo_fig, self.topo_ax = plt.subplots(1,1)
        self.cbar = None

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
    def update_topo_graph(self, data, first_time=False):
        self.topo_ax.clear()
        topo = self.create_topo(data, ["FZ","T8","T7"], "Delta band [0.4-5hz]",self.topo_ax,v_min=0.40,v_max=0.75)

        if first_time:
            self.cbar = create_cbar(self.topo_fig,topo, self.topo_ax, ylabel= "Relative power")

        self.topo_fig.canvas.draw()
        img = self.create_img_from_graph(self.topo_fig)

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
    def create_topo(data_frame, ch_to_plot, fig_title, ax, v_min=-0.022, v_max=0.022):
        from mne.viz import plot_topomap

        mask = np.array([True for _ in range(len(ch_to_plot))])

        locations = pd.read_csv('./channel_2d_location.csv', index_col=0)
        locations = locations.drop(index=["PO8", "PO7"])

        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                           linewidth=0, markersize=20)

        im, cn = plot_topomap(data_frame, locations.loc[ch_to_plot,['x', 'y']].values,
                              outlines='head', axes=ax, cmap='RdBu_r', show=False,
                              names=ch_to_plot, show_names=True,
                              mask=mask, mask_params=mask_params,
                              vmin=v_min, vmax=v_max, contours=7)
        ax.set_title(fig_title, fontsize=15)
        return im

    @staticmethod
    def create_img_from_graph(fig):
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
        img = cv2.resize(img, (640, 480))
        return img.astype(np.uint8)


def main():

    #Bleeding info
    # srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Juan\11-09-20\S01_T01_Bleeding")
    # start_time, end_time = 290.0, 290.0 + 75
    # task = "UJuan_S01_T01_VeinSutureBleeding_raw"

    #NoBleeding info
    srcPath = Path( r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Juan\11-09-20\S01_T01_NoBleeding")
    start_time, end_time = 375.0, 375.0 + 75
    task = "UJuan_S01_T01_VeinSutureNoBleeding_raw"

    srcPath = srcPath

    # Output video
    resultPath = Path("./Debug") / (task+".avi")
    frame_width = 640 * 2
    frame_height = 480
    out = cv2.VideoWriter(str(resultPath), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    # Open EEG and video ts files, and video
    eeg_file_name = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name))>0 ][0]
    print("loading eeg from {:}".format(eeg_file_name.name))
    eeg_file = pd.read_csv(eeg_file_name)

    ts_file = pd.read_csv(srcPath / "video_right_color_ts_trimmed.txt")
    cap = cv2.VideoCapture(str(srcPath / "video_right_color_trimmed.avi"))

    # create Graph generator
    graph_generator = GraphGenerator()
    topo_graph = graph_generator.update_topo_graph([0.4,0.6,0.5], first_time=True)

    #Forward Video to start time
    ts_file["ts_normalize"] = ts_file["ecm_ts"] - ts_file.iloc[0].loc["ecm_ts"]
    selected_frames = ts_file.loc[(ts_file["ts_normalize"] < start_time)]
    initial_frames_to_remove = selected_frames.shape[0]
    selected_frames = ts_file.loc[(ts_file["ts_normalize"] > start_time) & (ts_file["ts_normalize"] < end_time)]
    max_frames = selected_frames.shape[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frames_to_remove)  # skip all the initial frames

    count = 0
    prediction_array = []
    features_array = []
    raw_array = []
    prev_time = selected_frames.iloc[count]['ecm_ts'] - 0.005
    initial_ts = selected_frames.iloc[count]['ecm_ts']
    total_eeg_pt = 0
    window_length = 500
    window_overlap = 250

    feature_fact = feature_factory(2,1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()  # Capture frame-by-frame

            if ret and count < max_frames:
                # Get EEG data from file
                ts = selected_frames.iloc[count]['ecm_ts']
                data = eeg_file.loc[(eeg_file["COMPUTER_TIME"] > prev_time) & (eeg_file["COMPUTER_TIME"] < ts)]
                count_of_eeg = data.shape[0]
                total_eeg_pt += count_of_eeg
                prev_time = ts

                # Create plots
                graph_generator.update_eeg_data(data[EEG_channels].values)
                # graph_from_signals = (np.ones((480,640,3))*255).astype(np.uint8)
                # graph_from_signals = graph_generator.update_eeg_graph()  # np.ones((480,640,3))*255

                # Create graph for predictions
                if total_eeg_pt > window_overlap:
                    print("Make prediction")
                    new_chunk = graph_generator.eeg_data[:, -total_eeg_pt:][:, 0:window_overlap]
                    dataPts = feature_fact.make_prediction(new_chunk.transpose())
                    print(dataPts)

                    # graph_generator.update_predictions_data(prediction)
                    topo_graph = graph_generator.update_topo_graph(dataPts)
                    total_eeg_pt -= window_overlap

                    # Debug
                    prediction_array.append(dataPts)
                    # features_array.append(features)
                    # raw_array.append(raw)
                count += 1

                # print("time: {:f} total eeg points: {:d}".format(ts - initial_ts, total_eeg_pt))
                final_frame = np.hstack((frame, topo_graph,))
                out.write(final_frame)
                cv2.imshow('Frame', final_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    except Exception as e:
        print(e)
        print(traceback.print_exception())
    finally:
        # When everything done, release the video capture object
        cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        resultPath = Path("./Debug")
        pickle.dump(prediction_array, open(resultPath / '{:}_array_of_pred.pickle'.format(task), 'wb'))
        # pickle.dump(features_array, open('CheckPredictionPlot/{:}_array_of_features.pickle'.format(task), 'wb'))
        # pickle.dump(raw_array, open('CheckPredictionPlot/{:}_array_of_raw.pickle'.format(task), 'wb'))
        # pickle.dump(deep_model.dataBuffer, open('CheckPredictionPlot/{:}_data_buffer.pickle'.format(task), 'wb'))

if __name__ == "__main__":
    main()