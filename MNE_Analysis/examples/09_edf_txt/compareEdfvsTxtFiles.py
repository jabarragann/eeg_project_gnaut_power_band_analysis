import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]

if __name__ == "__main__":

    channel_to_show = 'CP5'
    file_txt = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-SurgicalTasks\txt\Jing\UJing_S01_T02_RunningSuture_raw.txt')
    txt_data = pd.read_csv(file_txt)
    txt_data = txt_data[EEG_channels]
    txt_data = txt_data[[channel_to_show]]

    file_edf = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-SurgicalTasks\edf\Jing\UJing_S01_T02_RunningSuture_raw.edf')
    raw = mne.io.read_raw_edf(file_edf)
    edf_data = raw.pick([channel_to_show]).get_data()


    fig, ax = plt.subplots(1,1)
    ax.plot(txt_data[0:5000])
    ax.plot(edf_data.squeeze()[0:5000] * 1e6)
    plt.show()