import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mne
import yasa
from BleedingTestV2.Utils import *

def main():
    mne.set_log_level("WARNING")
    path_dicts = {1:{"Blood"  : r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Juan\11-09-20\S01_T01_Bleeding\UJuan_S01_T01_VeinSutureBleeding_raw.txt",
                     "NoBlood": r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Juan\11-09-20\S01_T01_NoBleeding\UJuan_S01_T01_VeinSutureNoBleeding_raw.txt"}}
    path_dicts = path_dicts[1]

    start_times = {"Blood": 290.0, "NoBlood": 375.0}

    dict_results = {}
    index = pd.MultiIndex.from_product([["Blood", "NoBlood"], POWER_COEFFICIENTS],
                                       names=['condition', 'band'])
    means_results = pd.DataFrame(index= index, columns=EEG_channels)
    for k,p in path_dicts.items():
        dict_results[k] = {}
        path = Path(p)
        print(k,path.name,path.exists())

        raw = loadDataInRaw(path)

        window = 1.0
        overlap = 0.75
        epochs = splitDataIntoEpochs(raw, window, overlap) #epochs 2s, ovelap 1s: works fine

        #Crop to the specific segment
        seg_idx = np.where((epochs.events[:, 0] / 250 > start_times[k]) & (epochs.events[:, 0] / 250 < start_times[k] + 75))[0]
        epochs = epochs[seg_idx]

        ts, powerbands = calculatePowerBand(epochs,'na',window,'na')

        # # Crop to the specific segment
        # seg = ts.loc[(ts['ts'] > start_times[k]) & (ts['ts'] < start_times[k] + 75)]
        # powerbands = powerbands.loc[seg.index, :]

        meansFrame = powerbands.loc[:, pd.IndexSlice[:, :]].mean(axis=0)
        for band in POWER_COEFFICIENTS:
            means_results.loc[k,band] = meansFrame.loc[pd.IndexSlice[:, band]].values.reshape(1,30)

    means_results = means_results.astype(float)
    difference_between_conditions = means_results.loc["Blood"] - means_results.loc["NoBlood"]

    fig,axes =plt.subplots(1,4, figsize=(25,5))
    for a, band in enumerate(POWER_COEFFICIENTS):
        im = create_topo(difference_between_conditions.loc[band],band+" hard-easy",ax=axes[a],v_min=None,v_max=None)
        create_cbar(fig,im, axes[a], pad=0.25)

        if a == 3:
            break
    fig.tight_layout()
    fig.tight_layout()
    plt.show()
    x = 0

if __name__ == "__main__":
    main()