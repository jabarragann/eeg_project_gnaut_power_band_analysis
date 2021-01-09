import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def main():
    dataBlood = pickle.load(open("./Debug/UJuan_S01_T01_VeinSutureBleeding_raw_array_of_pred.pickle","rb"))
    dataBlood = np.array(dataBlood)
    print(dataBlood.shape)

    dataNoBlood = np.array(pickle.load(open("./Debug/UJuan_S01_T01_VeinSutureNoBleeding_raw_array_of_pred.pickle","rb")))

    fig, axes = plt.subplots(2,1)
    axes[0].plot(dataBlood[:,0],"-",color="blue", label="Fz",)
    axes[0].plot(dataNoBlood[:, 0],"--", color="blue", label="Fz")

    axes[1].plot(dataBlood[:, 1], "-", color="black", label="T8")
    axes[1].plot(dataNoBlood[:, 1],"--", color="black", label="T8")
    axes[1].axvline(14)

    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()