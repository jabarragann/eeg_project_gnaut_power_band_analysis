# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# Creating dataset
z = np.random.randint(100, size=(50))
x = np.random.randint(80, size=(50))
y = np.random.randint(60, size=(50))

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(x, y, z, color="green")
plt.title("simple 3D scatter plot")

# show plot
plt.show()