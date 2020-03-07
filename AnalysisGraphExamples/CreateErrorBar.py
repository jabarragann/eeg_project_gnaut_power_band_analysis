import matplotlib.pyplot as plt


x = [1, 2, 3, 4]
y = [1, 4, 10, 16]
e = [0.5, 1., 2, 2.]
plt.errorbar(x, y, yerr=e, fmt='o', linestyle='-', ecolor='black', capsize=5)
plt.show()