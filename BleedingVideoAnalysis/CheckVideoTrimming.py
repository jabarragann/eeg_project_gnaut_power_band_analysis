import cv2
import numpy as np
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\BleedingTests\Juan\11-09-20\S01_T01_NoBleeding")
    srcPath = srcPath

    #Open EEG and video ts files
    eeg_file = pd.read_csv(srcPath / "UJuan_S01_T01_VeinSutureNoBleeding_raw.txt")
    ts_file = pd.read_csv(srcPath / "video_right_color_ts_trimmed.txt")
    ts_file["ecm_ts"] = ts_file["ecm_ts"]

    #Open video
    cap = cv2.VideoCapture(str(srcPath / "video_right_color_trimmed.avi"))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frames in video {:d}, frames in ts file {:d}".format(total_frame_count, ts_file.shape[0]) )

    print("Initial EEG      {:f}".format(eeg_file.loc[0,"COMPUTER_TIME"]))
    print("Initial video ts {:f}".format(ts_file.loc[0,"ecm_ts"]))
    print("Final EEG      {:f}".format(eeg_file["COMPUTER_TIME"].values[-1]))
    print("Final video ts {:f}".format(ts_file["ecm_ts"].values[-1]))

