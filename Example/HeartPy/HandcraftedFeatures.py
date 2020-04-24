import heartpy as hp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np

#Import shimmer data
# df = pd.read_csv('./data/jackie_S1_T1_shimmer.txt')
# df = pd.read_csv('./data/juan_S2_T3_shimmer.txt')
# df = pd.read_csv('./data/ryan_S5_T1_shimmer.txt')
df = pd.read_csv('./data/jhony_S4_T5_shimmer.txt')

fs = 204.8
data, timer = df['PPG'].values[:int(fs*60)], df['ComputerTime'].values


data = hp.filter_signal(data, [0.8, 3.0], sample_rate=fs, order=3, filtertype='bandpass')

#minimum distance between two consecutive peaks 0.48s --> 100 samples at 204.8
peaks1, _   = find_peaks(data, height=0, distance = 100)
peaks = list(map(lambda x:[x,'p'],peaks1))
valleys1, _ = find_peaks(-1*data, height=0, distance = 100)
valleys = list(map(lambda x:[x,'v'],valleys1))
complete = peaks + valleys
complete = sorted(complete, key =lambda x:x[0])

fallTimes = []
riseTimes = []
for i in range(len(complete)):
    if i+1 == len(complete):
        break
    if complete[i][1] == 'p':
        if complete[i+1][1] == 'v':
            fallTimes.append(complete[i+1][0] - complete[i][0])
    elif complete[i][1]== 'v':
        if complete[i+1][1] == 'p':
            riseTimes.append(complete[i+1][0] - complete[i][0])

fallTimes = np.array(fallTimes)
riseTimes = np.array(riseTimes)
peaksHeight  =  np.array(list(map(lambda x: data[x[0]],peaks)))
valleyHeight =  np.array(list(map(lambda x: data[x[0]],valleys)))

features = [fallTimes, riseTimes, peaksHeight, valleyHeight]
labels = ['fallTimes', 'riseTimes', 'peaksHeight','valleyHeight']
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(features, labels=labels)
ax1.grid()

featuresDict = {'riseTimeStd':riseTimes.std(),
                'riseTimeMean':riseTimes.mean(),
                'fallTimeStd': fallTimes.std(),
                'riseTimeMean':fallTimes.mean(),
                'peaskStd':peaksHeight.std(),
                'valleysStd':valleyHeight.std()}

print(featuresDict)
fig2, ax2 = plt.subplots()
ax2.plot(data)
ax2.plot(peaks1, data[peaks1], "x")
ax2.plot(valleys1,data[valleys1],'o')
ax2.plot(np.zeros_like(data), "--", color="gray")
plt.show()