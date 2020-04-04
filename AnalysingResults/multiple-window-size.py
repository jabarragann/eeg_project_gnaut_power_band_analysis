import pandas as pd
import matplotlib.pyplot as plt

ticks = [0.6,0.7,0.8,0.9]

#Figure parameters
figParameters = {'top':0.946,
                'bottom':0.092,
                'left':0.094,
                'right':0.974,
                'hspace':0.45,
                'wspace':0.127}


df = pd.read_csv('multiple-window-size.csv', sep=',')

#Create plot
r,c = 5,2
fig, ax = plt.subplots(r,c,sharex='col', sharey='row')
plt.subplots_adjust(**figParameters)

#set common parameters
for i1 in range(r):
    for i2 in range(c):
        ax[i1,i2].set_ylim(0.5,1)
        ax[i1, i2].grid()

for i1, user in enumerate(['All','Jackie','Jhony','Juan','Ryan']):
    for i2,total in enumerate([60,90]):
        df1 = df.loc[(df['Total'] == total) & (df['User'] == user)]

        t = '{:}-{:0d}'.format(user,total) if user == 'All' else user
        color = 'red' if total==60 else 'blue'
        ax[i1, i2].set_title(t)

        x1 = df1['windowSize'].values
        y1 = df1['meanTest'].values
        e1 = df1['std'].values
        ax[i1, i2].errorbar(x1, y1, yerr=e1, fmt='o',color=color, linestyle='-', ecolor='black', capsize=5)
        ax[i1, i2].set_yticks(ticks=ticks)


plt.show()
