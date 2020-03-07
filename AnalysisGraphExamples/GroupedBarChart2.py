# libraries
import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.30

# set height of bar
beforeAcc = [12, 30, 1, 8]
afterAcc = [28, 6, 16, 5]

# Set position of bar on X axis
r1 = np.arange(len(beforeAcc))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, beforeAcc, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, afterAcc, color='#557f2d', width=barWidth, edgecolor='white', label='var2')

# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth - 0.1 for r in range(len(beforeAcc))], ['A', 'B', 'C', 'D'])

# Create legend & Show graphic
plt.legend()
plt.show()
