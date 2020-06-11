from pathlib import Path
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
from pyedflib import highlevel
import pyedflib


EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]

# General settings and file paths
mne.set_log_level("WARNING")

if __name__ == "__main__":

    srcPath = "C:/Users/asus/OneDrive - purdue.edu/RealtimeProject/Data/GNautilusInvertedTask_Pyprep"
    dstPath = "C:/Users/asus/OneDrive - purdue.edu/RealtimeProject/Data/GNautilusInvertedTask_Pyprep_edf"

    summaryFile = open("./summary.txt",'w')

    src = Path(srcPath)
    dst = Path(dstPath)

    for file in src.rglob("*.txt"):
        print("Processsing ",file.name)

        p2 = dst / file.parent.name
        user = file.parent.name
        destinationFile = dst / file.parent.name / file.with_suffix(".edf").name

        if not p2.exists():
            p2.mkdir(parents=True)

        sfreq = 250
        df = pd.read_csv(file)
        data = df[EEG_channels].values.transpose()

        # Create writer
        writer = pyedflib.EdfWriter(str(destinationFile), len(EEG_channels), file_type=1)

        # Create header
        writer.setPatientName(user)

        #Set label
        label = df['label'].values.mean()
        label = "low workload" if label < 7.5 else "high workload"
        writer.setPatientAdditional(label)

        # Signals
        signal_headers = highlevel.make_signal_headers(EEG_channels, sample_rate=250)
        writer.setSignalHeaders(signal_headers)
        writer.writeSamples(data)

        #close
        writer.close()

