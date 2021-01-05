import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def main():
    dataBlood = pickle.load(open("./Debug/UJuan_S01_T01_VeinSutureBleeding_raw_array_of_pred.pickle","rb"))
    dataBlood = np.array(dataBlood)
    print(dataBlood.shape)

    dataNoBlood = np.array(pickle.load(open("./Debug/UJuan_S01_T01_VeinSutureNoBleeding_raw_array_of_pred.pickle","rb")))

    plt.plot(dataBlood[:,0],"-",color="blue", label="Fz",)
    plt.plot(dataBlood[:,1],"-",color="black", label="T8")

    plt.plot(dataNoBlood[:, 0],"--", color="blue", label="Fz")
    plt.plot(dataNoBlood[:, 1],"--", color="black", label="T8")
    plt.axvline(14)

    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()