import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def getHeatMapData(path):
    dataSummary = {'Window Size': [], 'Lstm Sample Size': [], 'meanAcc': [], 'std': []}
    for file in path.glob('*.csv'):
        windowSize = int(re.findall('(?<=dow)[0-9]+(?=s)',file.name)[0][-2:])
        sampleSize = int(re.findall('(?<=Size)[0-9]+(?=s\.csv)',file.name)[0])

        if not windowSize == 50:
            #Load data
            df = pd.read_csv(file, sep = ',')
            meanAcc = df['TestAcc'].values.mean()
            std = df['TestAcc'].values.std()

            dataSummary['Window Size'].append(windowSize)
            dataSummary['Lstm Sample Size'].append(sampleSize)
            dataSummary['meanAcc'].append(meanAcc)
            dataSummary['std'].append(std)

            print(file.name, windowSize, sampleSize)

        summaryFrame = pd.DataFrame.from_dict(dataSummary)
        summaryFrame = pd.pivot_table(summaryFrame, values='meanAcc',index=['Lstm Sample Size'], columns='Window Size')

    return summaryFrame

def main():
    # Iterate through all the results files
    resultsDir = 'allChannelsExp-NoICA'
    path = Path('./').resolve().parent / 'results' / 'results_transfer9' / resultsDir
    summaryBaseline = getHeatMapData(path)

    resultsDir = 'fChannelsExp-NoICA'
    path = Path('./').resolve().parent / 'results' / 'results_transfer9' / resultsDir
    summaryFrame2 = getHeatMapData(path)

    comparison = summaryFrame2 - summaryBaseline

    f, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(str(resultsDir) + '- baseline(All channels noICA)' )
    sns.heatmap(comparison, cmap="PiYG", annot=True, fmt=".3%", linewidths=.5, ax=ax, vmin=+0.06, vmax=-0.12)
    plt.show()
    x = 0

if __name__ =='__main__':
    main()
