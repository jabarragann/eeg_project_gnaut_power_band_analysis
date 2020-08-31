import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

userList  = ['UI01','UI02','UI03','UI04','UI05','UI06']

if __name__ =='__main__':


    #Iterate through all the results files
    resultsDir = 'aa11a_deidentified_pyprep_reduced_complete_analysis/'
    path = Path('./').resolve().parent / 'results' / 'EegResults' /'results_transfer9' / resultsDir
    dataSummary = {'Window Size': [], 'Lstm Sample Size': [], 'meanAcc':[], 'std': []}

    for file in path.glob('*.csv'):
        windowSize = int(re.findall('(?<=dow)[0-9]+(?=s)', file.name)[0][-2:])
        sampleSize = int(re.findall('(?<=Size)[0-9]+(?=s\.csv)', file.name)[0])

        dfList = []
        if (windowSize == 10 and sampleSize == 160) or True:
            #Load data
            df = pd.read_csv(file, sep = ',')
            dfList.append(df)

    df = pd.concat(dfList, ignore_index=True)
    ax = sns.catplot(x="User", y="TestAcc", hue="TestSession", kind="bar", data=df)
    ax.set(ylim=(0.5, 1))
    plt.show()
    x = 0