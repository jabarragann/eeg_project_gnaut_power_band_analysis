from pathlib import Path

import yasa
from mne.viz import plot_topomap
from mne.time_frequency import psd_array_multitaper
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

if __name__ == "__main__":
    print("Topomap api")

    #Load locations
    locations = pd.read_csv('./channel_2d_location.csv')

    #load data
    file = Path('./../data/juan_S3_T2_epoc_pyprep.edf')
    raw = mne.io.read_raw_edf(file)
    mne.rename_channels(raw.info, renameChannels)
    raw = raw.set_montage('standard_1020')

    # Create epochs
    events_array = mne.make_fixed_length_events(raw, start=10, stop=None, duration=20)
    events_array = np.vstack((events_array, [67500 + 5000, 0, 1]))
    epochs = mne.Epochs(raw, events_array, tmin=-9.5, tmax=9.5)

    single_epoch = epochs[0]

    bd = yasa.bandpower(single_epoch.get_data()[0], sf=250, ch_names=raw.ch_names, win_sec=4,
                        bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                               (12, 30, 'Beta'), (30, 50, 'Gamma')],
                        bandpass=False, relative=True)

    fig, ax = plt.subplots(1,1)

    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=17)
    mask = np.full(32,True)
    mask[5:10] = True
    im, cn = plot_topomap(bd['Beta'].values, locations[['x','y']].values,
                          outlines='head', axes=ax,cmap='RdBu_r',show=False,
                          names = raw.ch_names, show_names=True,
                          mask=mask,mask_params=mask_params)

    fig.colorbar(im, ax=ax)
    # ax.legend()
    # plt.colorbar()
    plt.show()

    x=0

    # psd, freqs = psd_array_multitaper(single_epoch,250)

    # for c,c2  in zip(raw.ch_names,locations['ch_name']):
    #     print(c, c2)