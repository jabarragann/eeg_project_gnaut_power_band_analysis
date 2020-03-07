import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('transfer_session_data.csv', sep=',')
df  = df1.sort_values('user',ascending=True)

## Create lists for the plot
users = df['user'].values
x_pos = np.arange(len(users))
CTEs = df['test_acc_before'].values
error = df['std_test_acc_before'].values

# set width of bar
barWidth = 0.30

# set height of bar
beforeAcc = df['test_acc_before'].values
stdBefore = df['std_test_acc_before'].values
afterAcc = df['test_acc_after'].values
stdAfter = df['std_test_acc_after'].values

# Set position of bar on X axis
r1 = np.arange(len(beforeAcc))
r2 = [x + barWidth for x in r1]

# Make the plot
fig, ax = plt.subplots()
ax.set_title("Transfer Learning results - Frequency domain features")
ax.bar(r1, beforeAcc, yerr=stdBefore, color='#7f6d5f', width=barWidth, edgecolor='white', label='before transfer', capsize=10)
ax.bar(r2, afterAcc, yerr=stdAfter, color='#557f2d', width=barWidth, edgecolor='white', label='after transfer', capsize=10)

# Add xticks on the middle of the group bars
ax.set_xlabel('Users', fontweight='bold')
ax.set_ylabel('Test accuracy', fontweight='bold')

ax.set_xticks([r + barWidth - 0.1 for r in range(len(beforeAcc))])
ax.set_xticklabels(users,  fontweight='bold')
ax.yaxis.grid(True)
ax.set_ylim([0.4,1])

# Create legend & Show graphic
ax.legend()
plt.show()