from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import re
import numpy as np

def renameColumn(s):
    newStr = re.sub('-', '', s)
    return newStr

# df = pd.read_csv('./data.csv')
#
sessDict = {'UI01':'2','UI02':'1','UI03':'1','UI04':'1',
            'UI05':'1', 'UI06':'1', 'UI07':'1','UI08':'1'}

if __name__ ==  '__main__':

    dataRootPath = Path(r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\de-identified-pyprep-dataset-reduced-critically-exp")

    user = 'UI02'
    firstSession = sessDict[user]
    window = '02s'
    dataRootPath = dataRootPath / window / user

    dataList = []
    for fi in dataRootPath.rglob("*.txt"):
        sess = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', fi.name)[0]
        trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', fi.name)[0]

        if sess == firstSession:
            print(fi.name)
            df = pd.read_csv(fi, index_col=0)
            # df['Session'] = sess
            # df['Trial'] = trial
            # df['User'] = user
            dataList.append(df)

    df = pd.concat(dataList)

    #Remove unamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    features = df.columns.values
    features = df.columns.values[:-4]

    newDf = df[features[:16]]
    newDf['Label'] = df['Label'].values

    newDf.boxplot(by="Label", grid=False)
    plt.show()


