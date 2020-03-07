import numpy as np


def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
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
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

if __name__ == '__main__':

    #Sampling frequency
    sf = 100
    #Open txt file and get data
    with open('./data.txt','r') as f1:
        data = f1.readlines()
        data = list(map(lambda x:float(x),data))

    # Define the duration of the window to be 4 seconds
    win_sec = 4

    # Delta/beta ratio based on the absolute power
    lowCoeff = bandpower(data, sf, [0.0, 0.5], win_sec, relative=True)
    deltaCoeff = bandpower(data, sf, [0.5, 4], win_sec, relative=True)
    thetaCoeff = bandpower(data, sf, [4, 8], win_sec, relative=True)
    alphaCoeff = bandpower(data, sf, [8, 12], win_sec, relative=True)
    lowBetaCoeff =  bandpower(data, sf, [12, 16], win_sec, relative=True)
    highBetaCoeff = bandpower(data, sf, [16, 25], win_sec, relative=True)
    gammaCoeff =  bandpower(data, sf, [25, 50], win_sec, relative=True)
    total = [lowCoeff, deltaCoeff, thetaCoeff,alphaCoeff,lowBetaCoeff,highBetaCoeff,gammaCoeff]

    print("Low Relative power:       {:0.5f}".format(lowCoeff))
    print("Delta Relative power:     {:0.5f}".format(deltaCoeff))
    print("Theta Relative power:     {:0.5f}".format(thetaCoeff))
    print("Alpha Relative power:     {:0.5f}".format(alphaCoeff))
    print("Low Beta Relative power:  {:0.5f}".format(lowBetaCoeff))
    print("High Beta Relative power: {:0.5f}".format(highBetaCoeff))
    print("Gamma Relative power:     {:0.5f}".format(gammaCoeff))
    print("Total:                    {:0.5}".format(sum(total)))

