import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import simps

def plotData(x, time):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(time, x, lw=1.5, color='k')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Voltage')
    ax.set_xlim([time.min(), time.max()])
    ax.set_title('N3 sleep EEG data (F3)')
    plt.show()

    return

def plotSpectrum(psd, freqs):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(freqs, psd, color='k', lw=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density (V^2 / Hz)')
    ax.set_ylim([0, psd.max() * 1.1])
    ax.set_title("Welch's periodogram")
    ax.set_xlim([0, freqs.max()])
    plt.show()

    return


if __name__ == '__main__':

    #Open txt file and get data
    with open('./data.txt','r') as f1:
        data = f1.readlines()
        data = list(map(lambda x:float(x),data))

    #Define sampling frequency
    sf = 100. #Hz
    time = np.arange(len(data)) / sf


    # Define window length (4 seconds)
    win = 4 * sf

    # Calculate Welch Periodogram
    freqs, psd = signal.welch(data, sf, nperseg=win)

    # Find intersecting values in frequency vector (Delta coefficient)
    low, high = 0.5, 4
    idx_delta = np.logical_and(freqs >= low, freqs <= high)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

    # Compute the absolute power by approximating the area under the curve
    delta_power = simps(psd[idx_delta], dx=freq_res)
    print('Absolute delta power: %.3f uV^2' % delta_power)

    #Compute relative power (expressed as a percentage of total power)
    total_power = simps(psd, dx=freq_res)
    delta_rel_power = delta_power / total_power
    print('Relative delta power: %.5f' % delta_rel_power)


    # plotSpectrum(psd,freqs)

    # plotData(data,time)


