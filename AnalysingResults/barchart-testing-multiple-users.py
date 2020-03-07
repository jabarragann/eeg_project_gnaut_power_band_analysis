import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('barchart-testing-multiple-users.csv', sep=',')
df  = df1.sort_values('Users', ascending=True)

## Create lists for the plot
users = df['Users'].values
x_pos = np.arange(len(users))

# set width of bar
barWidth = 0.30

# set height of bar
meanAcc = df['MeanFinalAcc'].values
stdAcc = df['StdFinalAcc'].values


# Set position of bar on X axis
r1 = np.arange(len(meanAcc))
r2 = [x + barWidth for x in r1]

# Make the plot
fig, ax = plt.subplots()
ax.set_title("Cross-session results")
ax.bar(r1, meanAcc, yerr=stdAcc, color='#230aad', width=barWidth, edgecolor='white', label='before transfer', capsize=10)

# Add xticks on the middle of the group bars
ax.set_xlabel('Users', fontweight='bold')
ax.set_ylabel('Test accuracy', fontweight='bold')

ax.set_xticks([r + barWidth - 0.1 for r in range(len(meanAcc))])
ax.set_xticklabels(users,  fontweight='bold')
ax.yaxis.grid(True)
ax.set_ylim([0.5,1])

# Create legend & Show graphic
# ax.legend()
plt.show()