import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

userList  = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']

if __name__ =='__main__':

    f, axes = plt.subplots(2, 4, sharex='col', sharey='row')
    axes = axes.reshape(-1)

    plotConfig = {'top':0.88, 'bottom':0.08, 'left':0.11,'right':0.9,'hspace':0.5,'wspace':0.2}
    plt.subplots_adjust(**plotConfig)

    for user, ax in zip(userList,axes):
        #Iterate through all the results files
        resultsDir = 'aa11a_deidentified_pyprep_reduced_complete_analysis/'
        path = Path('./').resolve().parent / 'results' / 'EegResults' /'results_transfer9' / resultsDir
        dataSummary = {'Window Size': [], 'Lstm Sample Size': [], 'meanAcc':[], 'std': []}
        for file in path.glob('*.csv'):
            windowSize = int(re.findall('(?<=dow)[0-9]+(?=s)', file.name)[0][-2:])
            sampleSize = int(re.findall('(?<=Size)[0-9]+(?=s\.csv)', file.name)[0])

            #Load data
            df = pd.read_csv(file, sep = ',')
            df = df.loc[df['User']== user]
            meanAcc = df['TestAcc'].values.mean()
            std = df['TestAcc'].values.std()

            dataSummary['Window Size'].append(windowSize)
            dataSummary['Lstm Sample Size'].append(sampleSize)
            dataSummary['meanAcc'].append(meanAcc)
            dataSummary['std'].append(std)

            # print(file.name, windowSize, sampleSize)

        summaryFrame = pd.DataFrame.from_dict(dataSummary)
        summaryFrame = pd.pivot_table(summaryFrame, values='meanAcc',index=['Lstm Sample Size'], columns='Window Size')


        # ax.set_title("{:}-{:}".format(resultsDir,user))
        ax.set_title("{:}".format(user))
        sns.heatmap(summaryFrame, annot=True, fmt=".3", linewidths=.5, ax=ax)

    plt.show()
    x=0
