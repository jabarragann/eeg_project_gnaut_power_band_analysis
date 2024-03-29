"""
Use the following script to plot the results from the multi-user models or cross-user models.
This script shows what are the improvements of the accuracy as data of a new user is injected
into the model.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

# plt.style.use('seaborn-dark')

userNameMapping  = {'jackie':'Subject 1','ryan':'Subject 2', 'juan':'Subject 3',
                    'jhony' :'Subject 4','karuna':'Subject 5','santy':'Subject 6'}

def formatDataForGraph(path):
    # Format data
    df = pd.read_csv(path)
    proportions = df.proportionOfTransfer.unique()
    users = df.User.unique()
    proportions = list(map(lambda x: "{:0.3f}".format(x), proportions))
    # 0.00 proportion represents the accuracy of the model before transfer
    columns = ['User', 'TestSession', '0.000'] + proportions
    newFrame = pd.DataFrame(data=None, columns=columns)

    for u in users:
        temp = df.loc[df['User'] == u]
        userSessions = temp.TestSession.unique()
        for s in userSessions:
            print(u, s)
            temp = df.loc[(df['User'] == u) & (df['TestSession'] == s)]
            row = np.concatenate((np.array([u, s]), temp.TestAccBefore.values[0:1], temp.TestAccAfter.values))
            rowDf = pd.DataFrame(row.reshape(1, -1), columns=columns)
            newFrame = pd.concat([newFrame, rowDf], ignore_index=True)
    print(columns)

    return newFrame


def createBrokenAxisPlot():
    fig2, (ax1, ax2)  = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1]})
    # Broken Axis formating
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Change space between subplots
    subplotDict = {'top': 0.88, 'bottom': 0.135, 'left': 0.105, 'right': 0.955, 'hspace': 0.055, 'wspace': 0.2}
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


if __name__ == '__main__':

    # Load results data
    timePath = Path('C:\\Users\\asus\\PycharmProjects\\eeg_project_gnaut_power_band_analysis\\TimeClassification\\results\\results_transfer10\\aa11_pyprep\\window10s_sampleSize140s.csv')
    freqPath = Path(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\results\EegResults\results_transfer10\aa13a_pyprep_complete\window10s_sampleSize110s.csv')
    assert timePath.exists(), "Make sure time path is correctly set"
    assert freqPath.exists(), "Make sure freq path are correctly set"
    timeData = formatDataForGraph(timePath)
    freqData = formatDataForGraph(freqPath)


    # #Create Individual graphs for each user
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    # newFrame = timeData
    # nrow = 2
    # ncol = 4
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex="col")
    # axes = axes.reshape(-1)

    # for idx, u in enumerate(userNameMapping.keys()):
    #     axes[idx].set_title(userNameMapping[u])
    #     axes[idx].set_ylim((0.5,1))
    #     axes[idx].grid()
    #
    #     #Get data
    #     temp = newFrame.loc[newFrame['User'] == u]
    #     prop = temp.columns[2:]
    #     temp = temp[prop].values.astype(np.float)
    #     means = temp.mean(axis=0)
    #     std = temp.std(axis=0)
    #     x = list(map(lambda l: float(l), prop))
    #
    #
    #     #Plot data
    #     # axes[idx].errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=2,capthick=2,
    #     #                    alpha=0.5,color=colors[idx])
    #     axes[idx].errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=2, capthick=2,
    #                        alpha=0.5, color=colors[idx])
    #

    #Create summary graph of all users
    fig2, (ax1, ax2) = createBrokenAxisPlot()

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim([0.48, 1.0])  # outliers only
    ax2.set_ylim([0, 0.12])
    ax2.set_yticks([0, 0.10])

    #Title
    ax1.set_title("% of calibration data vs accuracy")
    #Common Y label
    fig2.text(0.03, 0.5, 'Average Accuracy', va='center', rotation='vertical')
    #X label
    ax2.set_xlabel("% of calibration data")
    #Grid
    ax1.grid(); ax2.grid()

    plotArg = dict(fmt='o', linestyle='-', capsize=2,capthick=2,alpha=0.5)
    for idx, (newFrame, label,errorColor) in  enumerate([(freqData,'Frequency domain model', 'black'), (timeData,'Time domain model', 'gray')]):
        #Create summary statistics
        prop = newFrame.columns[2:]
        temp = newFrame[prop].values.astype(np.float)
        means = temp.mean(axis=0)
        std = temp.std(axis=0)
        x = np.array(list(map(lambda l: float(l), prop)))
        ax1.errorbar((x+0.01*idx)*100, means,yerr=std, label=label, color=colors[idx], ecolor=errorColor, **plotArg) #yerr=std
        ax2.errorbar((x+0.01*idx)*100, means, label=label, color=colors[idx], ecolor=errorColor, **plotArg)
        ax2.set_xticks(x*100)

    ax1.set_facecolor('0.9');ax2.set_facecolor('0.9')
    ax2.legend( fancybox=True, shadow=True, )

    # ax2.legend() #Activate Legend
    plt.show()
