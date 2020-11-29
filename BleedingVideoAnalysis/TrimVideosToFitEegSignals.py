import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import re

if __name__ == "__main__":
    # srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\BleedingTests\Juan\11-09-20\S01_T01_NoBleeding")
    # srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\BleedingTests\Juan\11-09-20\S01_T01_Bleeding")

    # Open EEG and video ts files
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\TestsWithVideo\Idle-knot-typing-test\T03")
    eeg_file = [f for f in srcPath.rglob("*.txt") if len(re.findall("_S[0-9]+_T[0-9]+_", f.name)) > 0][0]
    eeg_file = pd.read_csv(eeg_file)
    ts_file = pd.read_csv(srcPath / "video_right_color_ts.txt")
    ts_file["ecm_ts"] = ts_file["ecm_ts"] * 1e-9

    # print("loading eeg from {:}".format(eeg_file))

    #Identify frames before starting signal and after ending
    frames_before = ts_file.loc[ts_file["ecm_ts"] < eeg_file.loc[0,"COMPUTER_TIME"]]
    initial_frames_to_remove = frames_before.shape[0]
    #Starting frames
    frames_after = ts_file.loc[ts_file["ecm_ts"] > eeg_file["COMPUTER_TIME"].values[-1]]
    ending_frames_to_remove =frames_after.shape[0]
    #Ending frames
    trimmed_ts_file = ts_file.loc[(ts_file["ecm_ts"] > eeg_file.loc[0,"COMPUTER_TIME"]) &
                                  (ts_file["ecm_ts"] < eeg_file["COMPUTER_TIME"].values[-1])]
    trimmed_ts_file.to_csv(srcPath / "video_right_color_ts_trimmed.txt")

    final_video_size = trimmed_ts_file.shape[0]
    print("Removed frames at the start {:d} and at the end {:d}".format(initial_frames_to_remove,ending_frames_to_remove))
    print("Final video size {:d}".format(final_video_size))

    #Open video
    cap = cv2.VideoCapture(str(srcPath / "video_right_color.avi"))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frames in video {:d}, frames in ts file {:d}".format(total_frame_count, ts_file.shape[0]) )
    cap.set(cv2.CAP_PROP_POS_FRAMES,initial_frames_to_remove) #skip all the initial frames

    # Output video
    frame_width = 640
    frame_height = 480
    out = cv2.VideoWriter(str(srcPath / "video_right_color_trimmed.avi"),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    # Check if camera opened successfully
    if not cap.isOpened():
      print("Error opening video stream or file")

    count = 0
    while cap.isOpened():
      ret, frame = cap.read() # Capture frame-by-frame
      if ret:
        out.write(frame)
        count += 1
        # cv2.imshow('Frame',frame)

        if count == final_video_size:
            break

        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break

      else:
        break

    cap.release()
    out.release()
    cv2.destroyAllWindows()