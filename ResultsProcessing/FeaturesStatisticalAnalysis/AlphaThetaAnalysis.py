from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import defaultdict

def renameColumn(s):
    newStr = re.sub('-', '', s)
    return newStr

# df = pd.read_csv('./data.csv')
#
sessDict = {'UI01':'2','UI02':'1','UI03':'1','UI04':'1',
            'UI05':'1', 'UI06':'1', 'UI07':'1','UI08':'1'}

if __name__ ==  '__main__':

    dataRootPath = Path(r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\de-identified-pyprep-dataset-reduced-critically-exp")

    window = '05s'
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
    finalResults = pd.DataFrame(columns = ['Channel','AlphaLow','AlphaHigh','AlphaDiff','AlphaPvalue',
                                     'ThetaLow','ThetaHigh','ThetaDiff','ThetaPvalue'])


    # "You need to remove '-' of the name to make the ordinary least squares work (OLS) model "
    df = df.rename(columns = renameColumn)
    # alphaThetaResults = pd.DataFrame(columns=['Channel', 'AlphaLow', 'AlphaHigh', 'AlphaDiff', 'AlphaPvalue',
    #                               'ThetaLow', 'ThetaHigh', 'ThetaDiff', 'ThetaPvalue'])
    # alphaThetaResults = defaultdict()

    for feat in features:
        ch, p = feat.split('-')

        if finalResults.loc[finalResults['Channel'] == ch].size == 0:
            finalResults = finalResults.append({"Channel": ch}, ignore_index=True)

        assert len(finalResults.loc[finalResults['Channel'] == ch].index.values == 1),  'Error with pandas logic'

        idx = finalResults.loc[finalResults['Channel'] == ch].index.values[0]
        if p == 'Alpha' or p == 'Theta' or p == 'Beta' or p == 'Delta':
            feat2 = renameColumn(feat)
            pivotT = df[[feat2]+['StrLabel']].groupby(['StrLabel']).mean()
            lowMean  = pivotT[feat2]['Low']
            highMean = pivotT[feat2]['High']

            # Ordinary Least Squares (OLS) model
            formula = '{:} ~ C(StrLabel) + C(User) + C(StrLabel):C(User)'.format(feat2)
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            # Add p-values
            anovaResults[ch][p] = anova_table["PR(>F)"]["C(StrLabel)"]

            # alphaThetaResults.loc['Channel'] = ch
            if p == 'Theta':
                finalResults.at[idx,"ThetaLow"] = lowMean
                finalResults.at[idx,'ThetaHigh'] = highMean
                finalResults.at[idx,'ThetaDiff'] = highMean - lowMean
                finalResults.at[idx,'ThetaPvalue'] = anova_table["PR(>F)"]["C(StrLabel)"]
            elif p == 'Alpha':
                finalResults.at[idx,'AlphaLow'] = lowMean
                finalResults.at[idx,'AlphaHigh'] = highMean
                finalResults.at[idx,'AlphaDiff'] = highMean - lowMean
                finalResults.at[idx,'AlphaPvalue'] = anova_table["PR(>F)"]["C(StrLabel)"]
            elif p == 'Beta':
                finalResults.at[idx, 'BetaLow'] = lowMean
                finalResults.at[idx, 'BetaHigh'] = highMean
                finalResults.at[idx, 'BetaDiff'] = highMean - lowMean
                finalResults.at[idx, 'BetaPvalue'] = anova_table["PR(>F)"]["C(StrLabel)"]
            elif  p == 'Delta':
                finalResults.at[idx, 'DeltaLow'] = lowMean
                finalResults.at[idx, 'DeltaHigh'] = highMean
                finalResults.at[idx, 'DeltaDiff'] = highMean - lowMean
                finalResults.at[idx, 'DeltaPvalue'] = anova_table["PR(>F)"]["C(StrLabel)"]

    finalResults.to_csv('./results/alpha_theta_analysis_{:}.csv'.format(window), sep=',', index=None)
