import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import re
import numpy as np

def renameColumn(s):
    newStr = re.sub('-', '', s)
    return newStr

if __name__ ==  '__main__':

    df = pd.read_csv('./data.csv')

    #Remove unamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    features = df.columns.values
    features = np.delete(df.columns.values, -1)

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
        model = ols('{:} ~ C(Label)'.format(feat2), data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        #Add p-values
        anovaResults[ch][p] = anova_table["PR(>F)"]["C(Label)"]

    anovaResults.to_csv('./p_values_results.csv', sep=',')
