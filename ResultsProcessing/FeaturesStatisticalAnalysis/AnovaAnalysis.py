import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import re

def renameColumn(s):
    newStr = re.sub('-', '', s)
    return newStr

if __name__ ==  '__main__':

    df = pd.read_csv('./data.csv')

    descriptor = df['Label']
    response = df['FP1-Delta']#.to_numpy()

    #show box plot
    boxPlot = df.boxplot(column='FP1-Delta', by='Label')
    # plt.show()

    # "You need to remove '-' of the name to make the ordinary least squares work (OLS) model "
    df = df.rename(columns = renameColumn)
    # Ordinary Least Squares (OLS) model
    model = ols('FP1Delta ~ C(Label)', data=df).fit()
    # model = ols('"FP1-Delta" ~ C(Label)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(anova_table)