from pathlib import Path
import mne
import re
import pandas as pd
import pickle
EEG_channels = ["FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                "PZ","P4","P8","PO3","PO4","OZ"]

def main():
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data")
    srcPath = srcPath / r"TestsWithVideo\Eyes-open-close-test\T01"

    # Open EEG file
    eeg_file = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name)) > 0][0]
    print("loading eeg from {:}".format(eeg_file))

    eeg_file = pd.read_csv(eeg_file)
    data = eeg_file[EEG_channels].values.transpose()
    data = data / 1e6 # Convert from uv to v
    ch_names = EEG_channels
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    scalings = {'eeg': 0.00003}
    raw.plot(n_channels=32, scalings=scalings, title='Auto-scaled Data from arrays',
             show=True, block=True)

if __name__ == "__main__":
    main()