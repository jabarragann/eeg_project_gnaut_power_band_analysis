import heartpy as hp
import matplotlib.pyplot as plt
import pandas as pd


#Import shimmer data
df = pd.read_csv('./data/juan_S7_T4_shimmer.txt')
data, timer = df['PPG'].values[:-1], df['ComputerTime'].values
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

# enhanced = hp.enhance_peaks(filtered, iterations=1)
#
# #and visualise
# plt.figure(figsize=(12,4))
# plt.plot(enhanced)
# plt.show()
# filtered = hp.filter_signal(data, cutoff=2, sample_rate=fs, order=3, filtertype='highpass')
# filtered = hp.filter_signal(data, cutoff = 4, sample_rate =fs, order = 5, filtertype='lowpass')
# scaled = hp.scale_data(filtered, lower=0, upper=1024)

#run the analysis
wd, m = hp.process(filtered, sample_rate = fs)

# wd, m = hp.process(filtered, sample_rate = fs)


#set large figure
plt.figure(figsize=(12,4))

#call plotter
hp.plotter(wd, m)

#display measures computed
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))