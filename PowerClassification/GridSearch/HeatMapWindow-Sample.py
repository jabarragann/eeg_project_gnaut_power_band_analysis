import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


if __name__ =='__main__':
    #'/temp/f-c-ChannelsExp-WithNoICA1'
    #Iterate through all the results files
    resultsDir = 'aa3_pyprep/'
    path = Path('./').resolve().parent / 'results' / 'EegResults' /'results_transfer9' / resultsDir
    dataSummary = {'Window Size': [], 'Lstm Sample Size': [], 'meanAcc':[], 'std': []}

    print(path)
    print(path.exists())
    for file in path.glob('*.csv'):
        windowSize = int(re.findall('(?<=dow)[0-9]+(?=s)',file.name)[0][-2:])
        sampleSize = int(re.findall('(?<=Size)[0-9]+(?=s\.csv)',file.name)[0])

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

    f, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(resultsDir)
    sns.heatmap(summaryFrame, annot=True, fmt=".3", linewidths=.5, ax=ax)
    plt.show()
    x=0
