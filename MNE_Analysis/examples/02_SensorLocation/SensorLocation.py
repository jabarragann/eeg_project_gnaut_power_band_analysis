import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne
import os

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
print(ten_twenty_montage)

fig = ten_twenty_montage.plot(kind='3d')
fig = ten_twenty_montage.plot(kind='topomap')
plt.show()