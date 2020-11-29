from pathlib import Path
import mne
import re
import pandas as pd
import matplotlib.pyplot as plt
from mne.filter import filter_data
import pickle

EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO3","PO4","OZ"]

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')

    return chName

def create_plot(orig_data, buffer_data):
    filtered = filter_data(orig_data, 250, 0.5, 30)

    fig, axes = plt.subplots(1)

    axes.plot(filtered[250 * 0:250 * 120], linewidth=0.5)
    axes.plot(orig_data[250 * 0:250 * 120], linewidth=0.5)
    axes.plot(buffer_data[250 * 0:250 * 120], linewidth=0.5)
    plt.show()
def main():
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data")
    srcPath = srcPath / r"TestsWithVideo\Eyes-open-close-test\T01"

    raw_array = pickle.load(open('../CheckPredictionPlot/data_buffer.pickle', 'rb'))

    # Open EEG file
    eeg_file = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name)) > 0][0]
    print("loading eeg from {:}".format(eeg_file))

    eeg_file = pd.read_csv(eeg_file)
    orig_data = eeg_file['FP1'].values
    create_plot(orig_data, raw_array[:,0])

if __name__ == "__main__":
    main()