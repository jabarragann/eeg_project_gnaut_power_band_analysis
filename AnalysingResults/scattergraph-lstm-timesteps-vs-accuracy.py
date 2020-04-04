import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('timesteps_data.csv', sep=',')

df1 = df.loc[df['user'] == 'jackie']
df2 = df.loc[df['user'] == 'juan']


fig, ax = plt.subplots(2,1,sharex=True)
x1 = df1['Timesteps'].values
y1 = df1['Average Acc'].values
e1 = df1['Std'].values
ax[0].set_title("Accuracy vs LSTM timesteps")
ax[0].errorbar(x1, y1, yerr=e1, fmt='o', linestyle='-', ecolor='black', capsize=5, label='jackie data' )
ax[0].set_ylim(0.6,1)
ax[0].legend()
# ax[0].set_xlabel("LSTM timesteps")
ax[0].set_ylabel("Testing acc")

x2 = df2['Timesteps'].values
y2 = df2['Average Acc'].values
e2 = df2['Std'].values
ax[1].set_xlabel("LSTM timesteps")
ax[1].set_ylabel("Testing acc")
ax[1].errorbar(x2, y2, yerr=e2, fmt='o', linestyle='-', ecolor='black', capsize=5, label='juan data', color='red')
ax[1].set_xticks(x2)
ax[1].set_ylim(0.6,1)
ax[1].legend()

plt.show()

print(df2)