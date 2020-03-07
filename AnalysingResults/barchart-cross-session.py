import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('transfer_session_data.csv', sep=',')


# # Create lists for the plot
users = df['user'].values
x_pos = np.arange(len(users))
CTEs = df['test_acc_before'].values
error = df['std_test_acc_before'].values


fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Average testing accuracy', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(users,  fontweight='bold')
ax.set_title('Cross-session classification results for all users.', fontweight='bold')
ax.yaxis.grid(True)
ax.set_ylim([0.5,1])

# Save the figure and show
plt.tight_layout()
plt.show()