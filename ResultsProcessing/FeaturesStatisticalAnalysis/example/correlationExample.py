import pingouin as pg
import pandas as pd


df = pd.read_csv('data_corr.csv')
print('%i subjects and %i columns' % df.shape)


corr = df.corr().round(2)

x = 0

#Plotting correlation matrix
from string import ascii_letters
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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