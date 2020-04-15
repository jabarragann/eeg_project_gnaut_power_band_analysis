import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')

# Grouped boxplot
f, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(x="day", y="total_bill", hue="smoker", data=df, palette="Set1",ax=ax)
plt.show()