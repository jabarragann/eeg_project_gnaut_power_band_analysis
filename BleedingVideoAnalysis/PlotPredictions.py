import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():
    pred_array = pickle.load(open('./CheckPredictionPlot/S03_T03_VeinLigationBlood_array_of_pred.pickle', 'rb'))
    pred_array = np.array(pred_array)

    fig, axes = plt.subplots(1, 1)
    colors = np.where(pred_array > 0.35, 'y', 'k')
    x = list(range(pred_array.shape[0]))
    axes.scatter(x, pred_array, c=colors)

    plt.show()

if __name__ == "__main__":
    main()

