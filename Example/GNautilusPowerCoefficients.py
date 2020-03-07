import yasa
import pandas as pd
import numpy as  np

EEG_labels = [  "COUNTER",
                "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ",
                "COMPUTER_TIME",
                "isThereMarker", "markerValue", "label", "5SecondWindow"]

#PO8 and PO7 channels were removed from the channel list because they are broken.
EEG_channels = [
                    "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                    "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                    "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                    "PZ","P4","P8","PO3","PO4","OZ"]

if __name__ == '__main__':

    #Sampling frequency
    sf = 250

    #Read data
    data = pd.read_csv('Juan_S6_T1_epoc.txt', sep=',')
    eegEvents = data["markerValue"].values

    #Get Initial index
    for i in range(0, len(eegEvents)):
        if eegEvents[i] == 'started':
            for j in range(i, len(eegEvents)):
                if eegEvents[j] == 'active' or eegEvents[j] == 'not_active':
                    initialIdx = j
                    break
            break
    # Get final index
    for i in range(initialIdx, len(eegEvents)):
        if eegEvents[i] == 'finished':
            finalIdx = i
            break

    data = data[initialIdx:finalIdx+1]
    totalTime = data['COMPUTER_TIME'].values[-1] -data['COMPUTER_TIME'].values[0]

    eegData = data[EEG_channels]
    eegData = eegData.transpose()

    data = eegData.values

    # Relative bandpower per channel on the whole recording (entire data)
    bd = yasa.bandpower(data, sf=sf, ch_names=EEG_channels, win_sec=4,
                        bands=[(0.0, 0.5, 'Low'), (0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                               (12, 30, 'Beta'), (30, 50, 'Gamma')])
    print("hello")