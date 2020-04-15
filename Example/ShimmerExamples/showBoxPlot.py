import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

complete = []
for user in ['jackie','ryan','jhony','juan']:
    root = Path('./').resolve() / '{:}features.csv'.format(user)
    df = pd.read_csv(root)
    complete.append(df)

df = pd.concat(complete)
df = df.dropna()

# Grouped boxplot
f, ax = plt.subplots(figsize=(5, 3))
sns.boxplot(x='user', y="bpm", hue="label", data=df, palette="Set1",ax=ax)
plt.show()