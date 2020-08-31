import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()


if __name__ =='__main__':
    #'/temp/f-c-ChannelsExp-WithNoICA1'
    #Iterate through all the results files
    # resultsDir = 'aa4_pyprep_lstm_20-140_W10-20/'
    resultsDir = 'aa11a_deidentified_pyprep_reduced_complete_analysis'
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
    # summaryFrame = pd.pivot_table(summaryFrame, values='meanAcc',index=['Lstm Sample Size'], columns='Window Size')
    summaryFrame.sort_values(by=['Window Size', 'Lstm Sample Size'], inplace=True)



    import matplotlib.pyplot as plt
    # plt.style.use('seaborn-dark')
    summaryFrame['Windows Size2'] = summaryFrame['Window Size'].astype(str)
    f, ax = plt.subplots(1,1,figsize=(9, 6))

    for w1 in [10]:
        tempFrame = summaryFrame.loc[summaryFrame['Window Size'] == w1]
        ax.plot(tempFrame["Lstm Sample Size"], tempFrame['meanAcc'],label=str(w1)+" sec", marker="*")
    ax.set_ylabel("Average prediction accuracy")
    ax.set_xlabel("Lstm sample size in seconds")
    ax.set_xticks(tempFrame["Lstm Sample Size"])
    ax.grid()
    ax.set_title('Hyperparameters Optimization')
    ax.legend(title="Window Size", fancybox = True, shadow=True, )
    ax.set_facecolor('0.9')

    # frame = legend.get_frame()  # sets up for color, edge, and transparency
    # frame.set_facecolor('#b4aeae')  # color of legend
    # frame.set_edgecolor('black')  # edge color of legend
    # frame.set_alpha(1)  # deals with transparency
    plt.show()

    # f, ax = plt.subplots(figsize=(9, 6))
    # ax.set_title(resultsDir)
    # sns.heatmap(summaryFrame, annot=True, fmt=".3", linewidths=.5, ax=ax)
    # plt.show()
    # x=0