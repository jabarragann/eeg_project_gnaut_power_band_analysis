from pyedflib import highlevel
import pandas as pd
import matplotlib.pyplot as plt

#read original file
EEG_channels = [
                    "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                    "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                    "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                    "PZ","P4","P8","PO3","PO4","OZ"]

#raw data information
user= "Jackie"
trial = '6'
dataframe = pd.read_csv('./raw/{:}_S1_T{:}_epoc.txt'.format(user,trial),delimiter=',')
raw_signals = dataframe[EEG_channels].values

# read processed signals from edf file
signals, signal_headers, header = highlevel.read_edf('./eeglab_processed/{:}_S1_T{:}_processed.edf'.format(user,trial))
print(signal_headers[0]['sample_rate']) # prints 256
processed_signals = signals.transpose()

#Why does the size of the signal change
processed_signals = processed_signals[:raw_signals.shape[0],:raw_signals.shape[1]]

#Create new dataframe
count = dataframe['COUNTER']
finalDataFrame = pd.DataFrame(data=processed_signals,columns=EEG_channels)
events = dataframe[['COMPUTER_TIME','isThereMarker','markerValue','label','5SecondWindow']]
result = pd.concat([count, finalDataFrame,events], axis=1)

result.to_csv('./eeglab_processed_csv/Jackie_S1_T{:}_eeglab.txt'.format(trial), index=None)


# plt.plot(result['FP1'])
# plt.show()
print()