import pandas as pd
import statsmodels.api as sm

if __name__ ==  '__main__':

    df = pd.read_csv('./data.csv')

    descriptor = df['Label']
    response = df['FP1-Delta']#.to_numpy()

    glm_binom = sm.GLM(response,descriptor, family=sm.families.Gaussian())
    res = glm_binom.fit()
    print(res.summary())

    x= 0