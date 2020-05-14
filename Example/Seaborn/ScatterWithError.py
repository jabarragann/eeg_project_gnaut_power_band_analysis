import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

fmri = sns.load_dataset("fmri")
# sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri, markers=True, dashes=False,)
sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            dashes=False, markers=True, kind="line", data=fmri)
plt.show()