import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def renameColumn(s):
    newStr = re.sub('-', '', s)
    return newStr

if __name__ ==  '__main__':

    df = pd.read_csv('./data.csv')

    #Remove unamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    features = df.columns.values
    features = np.delete(df.columns.values, -1)


    featDf = df[features]

    corr = featDf.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()