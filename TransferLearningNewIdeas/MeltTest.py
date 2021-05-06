import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
if __name__ == "__main__":

    df = pd.read_csv('./experiments/00_test/acc_df.csv',index_col=0)
    print(df.head())
    new_df = df.melt(id_vars=["rep", "condition"], var_name="type",value_name="acc")
    print(new_df.head(10))
    sns.boxplot(x='type',y='acc',hue='condition',data=new_df)
    plt.show()
