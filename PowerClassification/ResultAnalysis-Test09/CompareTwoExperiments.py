import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="whitegrid")


# experiment1Path = {"path":"aa16b_pyprep_complete/", "condition":"FullSet"}
# experiment2Path = {"path":"aa16_pyprep_complete/" , "condition":"SubSet"}


if __name__ =='__main__':

    experiment1Path = {"path":"aa14_pyprep_complete/", "condition":"FullSet"}
    experiment2Path = {"path":"aa15b_pyprep_complete/" , "condition":"SubSet"}

    windowToAnalyze = 20
    sizeToAnalyze = 150

    rootPath = Path('./').resolve().parent / 'results' / 'EegResults' /'results_transfer9'

    total = []
    for exp in [experiment1Path, experiment2Path]:
        p  = rootPath / exp['path']
        for file in p.glob('*.csv'):
            windowSize = int(re.findall('(?<=dow)[0-9]+(?=s)',file.name)[0][-2:])
            sampleSize = int(re.findall('(?<=Size)[0-9]+(?=s\.csv)',file.name)[0])

            if windowToAnalyze == windowSize and sizeToAnalyze == sampleSize:
                print(file.name, windowSize, sampleSize)
                #Load data
                df = pd.read_csv(file, sep = ',')
                df['condition'] = exp['condition']
                total.append(df)

    final = pd.concat(total)

    # ax = sns.boxplot(x="User", y="TestAcc", hue="condition",
    #                  data=final, palette="Set3")

    ax = sns.boxplot(y="TestAcc", x="condition",
                     data=final, palette="Set3")

    plt.show()
    x = 0