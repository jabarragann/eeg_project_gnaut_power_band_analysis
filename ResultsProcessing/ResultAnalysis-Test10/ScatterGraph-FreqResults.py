"""
Use the following script to plot the results from the multi-user models or cross-user models.
This script shows what are the improvements of the accuracy as data of a new user is injected
into the model.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn-dark')

if __name__ == '__main__':
    # userDict = {'ryan':'User 1', 'jhony':'User 2','juan':'User 3','jackie':'User 4'}
    #Format data
    df = pd.read_csv("./Data/window10s_sampleSize110s.csv")
    proportions = df.proportionOfTransfer.unique()
    users = df.User.unique()

    proportions = list(map(lambda x:"{:0.3f}".format(x),proportions))

    #0.00 proportion represents the accuracy of the model before transfer
    columns = ['User','TestSession','0.000'] + proportions

    newFrame = pd.DataFrame(data=None, columns=columns)

    for u in users:
        temp =  df.loc[df['User'] == u]
        userSessions = temp.TestSession.unique()
        for s in userSessions:
            print(u,s)
            temp = df.loc[(df['User'] == u) & (df['TestSession'] == s)]
            row = np.concatenate((np.array([u,s]),temp.TestAccBefore.values[0:1] ,temp.TestAccAfter.values))
            rowDf = pd.DataFrame(row.reshape(1,-1), columns=columns)
            newFrame = pd.concat([newFrame, rowDf], ignore_index=True)
    print(columns)

    #Create Individual graphs for each user
    nrow = 2
    ncol = 4
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex="col", sharey='row')
    axes = axes.reshape(-1)
    colors = ['blue','green','red','cyan','magenta','black','yellow','orange']

    fig.text(0.5, 0.04, 'Percentage of test user data', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Classification accuracy', va='center', rotation='vertical', fontsize=12)

    for idx, u in enumerate(users):
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
        axes[idx].errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=5,color=colors[idx])
        axes[idx].fill_between(x, means+std, means-std, facecolor=colors[idx], alpha=0.15)

    #Create summary graph of all users
    fig2, ax = plt.subplots(1,1)
    ax.set_title("All Users")
    ax.set_ylim((0.5,1))
    ax.grid()
    #Create summary statistics
    prop = newFrame.columns[2:]
    temp = newFrame[prop].values.astype(np.float)
    means = temp.mean(axis=0)
    std = temp.std(axis=0)
    x = list(map(lambda l: float(l), prop))
    ax.errorbar(x, means, yerr=std, fmt='o', linestyle='-', ecolor='black', capsize=5,color='orange')
    ax.fill_between(x, means + std, means - std, facecolor='orange', alpha=0.15)
    ax.set_xticks(x)
    ax.set_xlabel("Percentage of test user data")
    ax.set_ylabel("Classification Accuracy")

    plt.show()
