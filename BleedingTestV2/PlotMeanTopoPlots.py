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

        epochs = splitDataIntoEpochs(raw,2.0,1.0)
        ts, powerbands = calculatePowerBand(epochs,'na',2,'na')

        # Crop to the specific segment
        seg = ts.loc[(ts['ts'] > start_times[k]) & (ts['ts'] < start_times[k] + 75)]
        powerbands = powerbands.loc[seg.index, :]

        meansFrame = powerbands.loc[:, pd.IndexSlice[:, :]].mean(axis=0)
        for band in POWER_COEFFICIENTS:
            means_results.loc[k,band] = meansFrame.loc[pd.IndexSlice[:, band]].values.reshape(1,30)

    means_results = means_results.astype(float)
    difference_between_conditions = means_results.loc["Blood"] - means_results.loc["NoBlood"]

    fig,axes =plt.subplots(4,2, gridspec_kw={'width_ratios': [10,1]})
    for a, band in enumerate(POWER_COEFFICIENTS):
        im = create_topo(difference_between_conditions.loc[band],band+" hard-easy",ax=axes[a,0],v_min=None,v_max=None)
        create_cbar(fig,im, axes[a,1])
    fig.tight_layout()
    plt.show()
    x = 0

if __name__ == "__main__":
    main()