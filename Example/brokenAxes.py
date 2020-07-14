import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np

fig = plt.figure(figsize=(5,2))
bax = brokenaxes( ylims=((0, .2), (.5, 1)), hspace=.5)
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend()
bax.set_xlabel('time')
bax.set_ylabel('value')

plt.show()