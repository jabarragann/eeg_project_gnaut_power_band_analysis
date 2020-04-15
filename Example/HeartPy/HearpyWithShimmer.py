import heartpy as hp
import matplotlib.pyplot as plt
import pandas as pd


#Import shimmer data
# df = pd.read_csv('./data/jackie_S1_T1_shimmer.txt')
df = pd.read_csv('./data/juan_S2_T3_shimmer.txt')
# df = pd.read_csv('./data/ryan_S5_T1_shimmer.txt')
# df = pd.read_csv('./data/jhony_S4_T5_shimmer.txt')

data, timer = df['PPG'].values[:-50], df['ComputerTime'].values
fs = 204.8

#and visualise
plt.figure(figsize=(12,4))
plt.plot(data)
plt.show()

#Filter signal
# filtered = hp.filter_signal(data, cutoff = 3, sample_rate = fs, order = 5, filtertype='lowpass')
filtered = hp.filter_signal(data, [0.7, 3.5], sample_rate=fs, order=3, filtertype='bandpass')

#and visualise
plt.figure(figsize=(12,4))
plt.plot(filtered)
plt.show()


#run the analysis
wd, m = hp.process(filtered, sample_rate = fs, calc_freq=True)

# wd, m = hp.process(filtered, sample_rate = fs)


#set large figure
plt.figure(figsize=(12,4))

#call plotter
hp.plotter(wd, m)

#display measures computed
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))