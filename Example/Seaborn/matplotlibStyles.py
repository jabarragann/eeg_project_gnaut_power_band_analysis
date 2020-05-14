import matplotlib.pyplot as plt
import numpy as np

print(plt.style.available)
# plt.style.use([ 'seaborn-dark'])
plt.style.use([ 'seaborn-ticks','seaborn-dark',])

plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
plt.grid()

for style in plt.style.available:
    print(style)
    with plt.style.context([style]):
        plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
        plt.grid()
    plt.show()