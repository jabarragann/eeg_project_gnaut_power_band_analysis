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

    window = '02s'
    dataList = []
    for u in sessDict.keys():
        for fi in (dataRootPath / window / u).rglob("*.txt"):
            sess = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', fi.name)[0]
            trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', fi.name)[0]

            if sess == sessDict[u]:
                print(fi.name)
                df = pd.read_csv(fi, index_col=0)
                df['Session'] = sessDict[u]
                df['Trial'] = trial
                df['User'] = u
                dataList.append(df)

    df = pd.concat(dataList)

    #Create string labels
    df['StrLabel'] = df['Label'].map(lambda x: "High" if x == 1 else "Low")
    #Remove unamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    features = df.columns.values
    features = df.columns.values[:-5]

    channels = set()
    bandpower = set()

    for i in features:
        ch, p = i.split('-')
        channels.add(ch); bandpower.add(p);

    anovaResults =  pd.DataFrame(np.zeros((len(bandpower), len(channels))), columns=channels, index=bandpower)
    x = 0

    # "You need to remove '-' of the name to make the ordinary least squares work (OLS) model "
    df = df.rename(columns = renameColumn)
    # features = list(map(renameColumn,features))

    for feat in features:
        ch, p = feat.split('-')
        feat2 = renameColumn(feat)

        # Ordinary Least Squares (OLS) model
        formula = '{:} ~ C(StrLabel) + C(User) + C(StrLabel):C(User)'.format(feat2)
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        #Add p-values
        anovaResults[ch][p] = anova_table["PR(>F)"]["C(StrLabel)"]

    anovaResults.to_csv('./p_values_all_users.csv', sep=',')
