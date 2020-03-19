import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Users,MeanBefore,StdBefore,MeanAfter,StdAfter,MeanIncrement
df = pd.read_csv('barchart-two-conditions.csv', sep=',')
#df  = df1.sort_values('Users',ascending=True)

## Create lists for the plot
conditions = df['conditions']
x_pos = np.arange(len(conditions))
beforeAcc = df['simple mean'].values
stdBefore = df['simple std'].values
afterAcc = df['comp mean'].values
stdAfter = df['comp std'].values

# set width of bar
barWidth = 0.30

# Set position of bar on X axis
r1 = np.arange(len(beforeAcc))
r2 = [x + barWidth for x in r1]

# Make the plot
fig, ax = plt.subplots()
ax.set_title("comparison between simple vs optimized model")
ax.bar(r1, beforeAcc, yerr=stdBefore, color='#a64403', width=barWidth, edgecolor='white', label='Simple', capsize=10)
ax.bar(r2, afterAcc, yerr=stdAfter, color='#0d8c0f', width=barWidth, edgecolor='white', label='optimized', capsize=10)

# Add xticks on the middle of the group bars
ax.set_xlabel('Users', fontweight='bold')
ax.set_ylabel('Test accuracy', fontweight='bold')

ax.set_xticks([r + barWidth - 0.1 for r in range(len(beforeAcc))])
ax.set_xticklabels(conditions,  fontweight='bold')
ax.yaxis.grid(True)
ax.set_ylim([0.5,0.9])

# Create legend & Show graphic
ax.legend()
plt.show()