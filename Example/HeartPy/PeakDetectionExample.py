import heartpy as hp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np

#Import shimmer data
df = pd.read_csv('./data/jackie_S1_T1_shimmer.txt')
# df = pd.read_csv('./data/juan_S2_T3_shimmer.txt')
# df = pd.read_csv('./data/ryan_S5_T1_shimmer.txt')
# df = pd.read_csv('./data/jhony_S4_T5_shimmer.txt')

fs = 204.8
b = int(fs*30)*3
e = int(fs*30)*4
data, timer = df['PPG'].values[b:e], df['ComputerTime'].values

plt.plot(data)
plt.show()

data = hp.filter_signal(data, [0.8, 3.0], sample_rate=fs, order=3, filtertype='bandpass')

peaks, _   = find_peaks(data, height=0, distance = 100)
valleys, _ = find_peaks(-1*data, height=0, distance = 100)

plt.plot(data)
plt.plot(peaks, data[peaks], "x")
plt.plot(valleys,data[valleys],'o')
plt.plot(np.zeros_like(data), "--", color="gray")
plt.show()