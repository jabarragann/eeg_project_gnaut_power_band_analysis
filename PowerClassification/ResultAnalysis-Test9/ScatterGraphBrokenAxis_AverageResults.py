import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()


def createBrokenAxisPlot():
    fig2, (ax1, ax2)  = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # Broken Axis formating
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Change space between subplots
    subplotDict = {'top': 0.88, 'bottom': 0.170, 'left': 0.120, 'right': 0.955, 'hspace': 0.055, 'wspace': 0.2}
    fig2.subplots_adjust(**subplotDict)

    # Draw diagonal lines in the axes coordinates
    d = .008  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)  # arguments to pass to plot
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d * 6, 1 + d * 6), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d * 6, 1 + d * 6), **kwargs)  # bottom-right diagonal

    return fig2, (ax1, ax2)

if __name__ =='__main__':
    #'/temp/f-c-ChannelsExp-WithNoICA1'
    #Iterate through all the results files
    resultsDir = 'aa11a_deidentified_pyprep_reduced_complete_analysis//'
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
    # f, ax = plt.subplots(1,1,figsize=(9, 6))

    # Create broken axis graph
    fig2, (ax1, ax2) = createBrokenAxisPlot()

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim([0.58, 0.9])  # outliers only
    ax2.set_ylim([0, 0.12])
    ax2.set_yticks([0, 0.10])

    #Axes parameters
    fig2.text(0.03, 0.5, 'Average prediction accuracy', va='center', rotation='vertical') # Common Y label
    ax2.set_xlabel("Lstm sample size in seconds")

    ax1.grid(); ax2.grid()
    ax1.set_title('Hyperparameters Optimization')

    plotArg = dict(marker='o', linestyle='-', alpha=0.5)
    for w1 in [10,20]:
        tempFrame = summaryFrame.loc[summaryFrame['Window Size'] == w1]
        ax1.plot(tempFrame["Lstm Sample Size"], tempFrame['meanAcc'],label=str(w1)+" sec", **plotArg)
        ax2.plot(tempFrame["Lstm Sample Size"], tempFrame['meanAcc'], label=str(w1) + " sec", **plotArg)

    ax2.set_xticks(tempFrame["Lstm Sample Size"])
    ax1.legend(title="Window Size", fancybox = True, shadow=True, )
    ax1.set_facecolor('0.9'); ax2.set_facecolor('0.9')

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