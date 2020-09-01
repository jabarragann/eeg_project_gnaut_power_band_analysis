"""
Use the following script to plot the results from the multi-user models or cross-user models.
This script shows what are the improvements of the accuracy as data of a new user is injected
into the model.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


plt.style.use('seaborn-dark')
USERS = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']

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
if __name__ == '__main__':

    #Load results data
    timePath = Path('C:\\Users\\asus\\PycharmProjects\\eeg_project_gnaut_power_band_analysis\\TimeClassification\\results\\results_transfer10\\aa11_pyprep\\window10s_sampleSize140s.csv')
    freqPath = Path(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\results\EegResults\results_transfer10\aa11_pyprep\window10s_sampleSize110s.csv')
    assert timePath.exists(), "Make sure time path is correctly set"
    assert freqPath.exists(), "Make sure freq path are correctly set"
    timeData = formatDataForGraph(timePath)
    freqData = formatDataForGraph(freqPath)

    #Plot individual user results
    framesDict = {'Time domain data results': timeData, 'Frequency domain models results': freqData}
    for label, newFrame in framesDict.items():
        #Create Individual graphs for each user
        nrow = 2
        ncol = 4
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex="col")
        fig.suptitle(label)
        axes = axes.reshape(-1)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'orange']

        for idx, u in enumerate(USERS):
            axes[idx].set_title(u)
            axes[idx].set_ylim((0.5,1))
            axes[idx].grid()

            #Get data
            temp = newFrame.loc[newFrame['User'] == u]
            prop = temp.columns[2:]
            temp = temp[prop].values.astype(np.float)
            means = temp.mean(axis=0)
            std = temp.std(axis=0)
            x = list(map(lambda l: float(l), prop))

            #Plot data
            # axes[idx].errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=2,capthick=2,
            #                    alpha=0.5,color=colors[idx])
            axes[idx].errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=2, capthick=2,
                               alpha=0.5, color=colors[idx])

    #Create summary graph of all users
    fig2, ax = plt.subplots(1,1)
    ax.set_title("All Users average")
    ax.set_ylim((0.5,1))
    ax.grid()

    errorColor = ['black','gray']
    for idx, (newFrame, label) in  enumerate([(freqData,'Bandpower model'), (timeData,'Time domain model')]):
        #Create summary statistics
        prop = newFrame.columns[2:]
        temp = newFrame[prop].values.astype(np.float)
        means = temp.mean(axis=0)
        std = temp.std(axis=0)
        x = np.array(list(map(lambda l: float(l), prop)))
        ax.errorbar((x+0.01*idx)*100, means,yerr=std, fmt='o', linestyle='-', ecolor=errorColor[idx], capsize=2,capthick=2,
                     alpha=0.6, label=label, color=colors[idx])

        # ax.errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=2, capthick=2,
        #             alpha=0.5, color=colors[idx])
        ax.set_xticks(x*100)
        ax.set_xlabel("% of calibration data of new user")
        ax.set_ylabel("Average Prediction Accuracy")

    ax.legend()
    plt.show()
