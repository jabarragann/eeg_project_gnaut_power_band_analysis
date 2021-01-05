import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mne
import yasa

EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO3","PO4","OZ"]
POWER_COEFFICIENTS = ["Delta","Theta","Alpha", "Beta"]

def clean_axes(a):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.set_xticks([])
    a.set_xticks([],minor=True)
    a.set_yticks([])
    a.set_yticks([],minor=True)

def create_cbar(fig, im, ax, ylabel="default"):
    cbar = fig.colorbar(im, ax=ax, pad=0.20, fraction=0.046,)
    clean_axes(ax)
    cbar.ax.set_ylabel(ylabel, rotation=270, fontsize=12, labelpad=22)
    cbar.ax.yaxis.set_ticks_position('left')
    return cbar

def create_topo(data_frame, fig_title, ax, v_min=-0.022, v_max=0.022):
    from mne.viz import plot_topomap

    mask = np.array([True for i in range(30)])

    locations = pd.read_csv('./channel_2d_location.csv', index_col=0)
    locations = locations.drop(index=["PO8", "PO7"])

    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                       linewidth=0, markersize=6)

    im, cn = plot_topomap(data_frame, locations[['x', 'y']].values,
                          outlines='head', axes=ax, cmap='jet', show=False,
                          names=data_frame.index, show_names=True,
                          mask=mask, mask_params=mask_params,
                          vmin=v_min, vmax=v_max, contours=7)
    ax.set_title(fig_title, fontsize=15)
    return im

def splitDataIntoEpochs(raw, frameDuration, overlap):
    w1 = frameDuration
    sf = 250

    eTime = int(w1 / 2 * sf) + raw.first_samp
    events_array = []
    while eTime < raw.last_samp:
        events_array.append([eTime, 0, 1])
        eTime += sf * (w1 - overlap)

    events_array = np.array(events_array).astype(np.int)
    epochs = mne.Epochs(raw, events_array, tmin=-(w1 / 2), tmax=(w1 / 2))

    return epochs

def calculatePowerBand(epoched_data, data_label, window,trial, sf=250):
    counter=0
    dataDict = {}
    epoched_data.load_data()

    if window < 4:
        win_sec = window * 0.95
    else:
        win_sec = 4

    # Power Band dataset
    # multi_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['ts', 'row'])
    columns = pd.MultiIndex.from_product([EEG_channels, POWER_COEFFICIENTS],
                                         names=['subject', 'type'])
    # create the DataFrame
    powerBandDataset = pd.DataFrame(columns=columns)
    timestamps = pd.DataFrame(columns=['ts'])

    for i in range(len(epoched_data)):
        data  =  epoched_data[i]
        data = data.get_data().squeeze()
        data  *= 1e6

        bd = yasa.bandpower(data, sf=sf, ch_names=EEG_channels, win_sec=win_sec,
                            bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                   (12, 30, 'Beta'), (30, 50, 'Gamma')])


        # # Reshape coefficients into a single row vector
        # bd = bd[POWER_COEFFICIENTS].values.reshape(1, -1)

        # Create row name, label and add to data dict
        rowName = 'T' + str(trial) + '_' + str(counter)
        #Update counter
        counter+=1
        powerBandDataset.loc[rowName] = bd[POWER_COEFFICIENTS].values.reshape(-1)
        timestamps.loc[rowName] = epoched_data[i].events[0,0] / 250

        # bd = np.concatenate((bd, np.array([data_label]).reshape(1, -1)), axis=1)
        # dataDict[rowName] = np.squeeze(bd)

    return timestamps, powerBandDataset

def loadDataInRaw(path, sfreq=250):
    # Load data
    df = pd.read_csv(path)
    data = df[EEG_channels].values.transpose()
    # Create MNE object
    ch_names = EEG_channels
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data / 1e6, info)

    return raw