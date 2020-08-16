import numpy as np
import mne
import pandas as pd
from pathlib import Path
import re


if __name__ == '__main__':
    mne.set_log_level("WARNING")

    df = pd.DataFrame( columns=['uid','session','trial','task','preprocess','recording time(s)','recording time(min)'])

    path = Path('C:\\Users\\asus\\OneDrive - purdue.edu\\RealtimeProject\\Experiment1-Pilot').resolve()
    for idx, file in enumerate(path.rglob('*.edf')):

        # Rename files --> remove identifiers
        uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
        session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
        trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
        task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
        preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

        if preprocess == 'pyprep':
            print(idx, uid, session, trial, task, preprocess)

            raw = mne.io.read_raw_edf(file)
            # raw.plot(block=True)

            recordingTime = raw.times.max()

            df2 = pd.DataFrame([[uid,session,trial,task,preprocess, recordingTime,recordingTime/60]],
                               columns=['uid', 'session', 'trial', 'task', 'preprocess', 'recording time(s)', 'recording time(min)'])
            df = df.append(df2, ignore_index=True)


    df.to_csv('dataset_stats.csv', sep=',')
