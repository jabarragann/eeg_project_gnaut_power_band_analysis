import heartpy as hp
import matplotlib.pyplot as plt
import pandas as pd


#Import shimmer data
df = pd.read_csv('./data/jackie_S7_T4_shimmer.txt')
data, timer = df['PPG'].values[:5120], df['ComputerTime'].values
fs = 204.8

#Filter signal
# filtered = hp.filter_signal(data, cutoff=2, sample_rate=fs, order=3, filtertype='highpass')
# filtered = hp.filter_signal(data, cutoff = 4, sample_rate =fs, order = 5, filtertype='lowpass')
# scaled = hp.scale_data(filtered, lower=0, upper=1024)

#and visualise
plt.figure(figsize=(12,4))
plt.plot(data)
plt.show()

#run the analysis
wd, m = hp.process(data, sample_rate = fs)


#set large figure
plt.figure(figsize=(12,4))

#call plotter
hp.plotter(wd, m)

#display measures computed
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))