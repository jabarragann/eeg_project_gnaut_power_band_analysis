from pathlib import Path
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

#print("[{:}]".format(','.join(['"'+val+'"' for val in dataContainer.columns.values])))
#["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50", "hr_mad", "sd1", "sd2", "s", "sd1/sd2"]
if __name__== '__main__':

    metrics = ["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50", "hr_mad", "sd1", "sd2", "s", "sd1/sd2"]
    DATA_DIR = Path('./').resolve().parent / 'data' / 'shimmerPreprocessed' / '60s'
    print('Data Directory')
    print(DATA_DIR)

    dataContainer = []
    for file in DATA_DIR.rglob('*.txt'):
        df = pd.read_csv(file,sep=',')
        df['user'] = file.parent.name
        dataContainer.append(copy.deepcopy(df))

    dataContainer = pd.concat(dataContainer)
    counter=0
    f, ax = plt.subplots(6,2,figsize=(10, 6),sharex='col')
    for r in range(6):
        for c in range(2):
            g = sns.boxplot(x='user', y=metrics[counter],
                            hue="label", data=dataContainer,
                            palette="Set1", ax=ax[r,c])
            g.legend().remove()
            counter += 1
    g.legend()
    plt.show()

