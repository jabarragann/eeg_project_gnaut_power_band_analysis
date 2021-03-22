import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


if __name__ == "__main__":

    data = pd.read_csv('./test_data.txt')
    data = data.reset_index()

    fig, axes = plt.subplots(2,2, sharex=True,sharey=True)
    axes = axes.reshape(-1)
    for u,ax in zip(['U1','U2','U3','U4'],axes):
        data_2 = data.loc[(data['user'] == u) & ~(data['passes_count'] == -1)]
        for c in ['normal','inverted']:
            df = data_2.loc[data['condition']==c]
            x = df['counter']
            y = df['mean_errors'] * df['passes_count']
            ax.plot(x,y, '*-', markersize=13, label=c)
            ax.set_xticks(range(1,19))
            ax.set_xlabel("Number of trials done")
            ax.set_ylabel("Number of drops")
            ax.set_title("{} performance".format(df['user'].values[0]))
        ax.grid()
        ax.axvspan(1, 6,   alpha=0.1, color='red')
        ax.axvspan(7, 12,  alpha=0.1, color='green')
        ax.axvspan(13, 18, alpha=0.1, color='blue')
        ax.legend()

    plt.show()