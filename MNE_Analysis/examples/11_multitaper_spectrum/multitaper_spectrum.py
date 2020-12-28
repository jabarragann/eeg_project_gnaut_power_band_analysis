import yasa
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

mne.set_log_level("WARNING")


def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0, n_jobs=4)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def multitaper_bandpower(data,sf, eeg_channels, bands=None, relative=True,n_jobs=5):
    if bands is None:
        bands=[(0.5, 4, 'Delta'),(4, 8, 'Theta'),
                 (8, 12, 'Alpha'), (12, 30, 'Beta'),
                 (30, 50, 'Gamma')]

    psd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0, n_jobs=n_jobs)

    freq_res = freqs[1] - freqs[0]  # Frequency resolution

    bandpower_df = pd.DataFrame(columns=eeg_channels).astype(float)
    for ch_idx,ch  in enumerate(eeg_channels):
        for lf,hf,bn in bands:
            # Find index of band in frequency vector
            idx_band = np.logical_and(freqs >= lf, freqs <= hf)
            # Integral approximation of the spectrum using parabola (Simpson's rule)
            bp = simps(psd[ch_idx,idx_band], dx=freq_res)
            bandpower_df.loc[bn, ch] = bp

        total_power = simps(psd[ch_idx, :], dx=freq_res)
        if relative:
            bandpower_df[ch] = bandpower_df[ch] / total_power
        bandpower_df.loc['TotalAbsPow', ch] = total_power

    return bandpower_df.transpose()


def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

if __name__ == "__main__":

    #Read eeg file
    file = Path('./../data/juan_S3_T2_epoc_pyprep.edf')
    raw = mne.io.read_raw_edf(file)

    #Rename Channel
    mne.rename_channels(raw.info, renameChannels)
    #Set montage (3d electrode location)
    raw = raw.set_montage('standard_1020')

    #Create events every 20 seconds
    events_array = mne.make_fixed_length_events(raw, start=10, stop=None, duration=10)

    #Get 20 seconds Epochs from data
    epochs = mne.Epochs(raw, events_array, tmin=-4.5, tmax=4.5)
    # epochs.plot(n_epochs=4)
    epochs = epochs.get_data() * 1e6

    bd_taper = multitaper_bandpower(epochs[10],250,raw.ch_names)

    bd_yasa = yasa.bandpower(epochs[15], sf=250, ch_names=raw.ch_names, win_sec=4,
                        bands=[(0.5, 4, 'Delta'),(4, 8, 'Theta'), (8, 12, 'Alpha'),
                               (12, 30, 'Beta'), (30, 50, 'Gamma')],
                        bandpass=False, relative=True)

    b ='Theta'
    x,y = bd_yasa[b].values, bd_taper[b].values
    plt.plot(x,y, 'ok')
    r = np.corrcoef(x, y)
    print(r)
    plt.show()

    # print(epochs[6,2,:].shape)
    # psd, freqs = psd_array_multitaper(epochs[6,:,:], 250, adaptive=True, \
    #                                   normalization='full', verbose=0, n_jobs=6)
    #
    # freqs_w, psd_w = welch(epochs[6,8,:], 250, nperseg=int(250*4.5))
    #
    # print(psd.shape,psd_w.shape)
    # print(freqs_w.shape)

    # # plot different spectrum types:
    # fig, ax = plt.subplots(3,1)
    #
    # ax[0].set_title("multitaper Spectrum")
    # ax[0].plot(freqs, psd[8,:], '-k')
    #
    # ax[1].set_title("welch Spectrum")
    # ax[1].plot(freqs_w, psd_w, '-k')
    #
    # ax[2].set_title("Both")
    # ax[2].plot(freqs, psd[8,:], label="taper")
    # ax[2].plot(freqs_w, psd_w, label="welch" )
    # ax[2].legend()
    # plt.show()
