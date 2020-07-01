import numpy as np
import mne
import pandas as pd
from pathlib import Path


#Read eeg file
file = Path('./../data/juan_S3_T2_epoc_pyprep.edf')
raw = mne.io.read_raw_edf(file)

#Create events every 15 seconds
events_array = mne.make_fixed_length_events(raw, start=0, stop=300, duration=15)

scalings = {'eeg': 'auto'}
raw.plot(n_channels=32, scalings=scalings, title='Edf sample', show=True, block=True,
         events=events_array,event_color={1:'r'}, duration=80.0)
