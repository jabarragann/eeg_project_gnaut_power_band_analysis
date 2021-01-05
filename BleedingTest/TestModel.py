
from PowerClassification.Utils.NetworkTraining import DataLoaderModule
from PowerClassification.Utils.NetworkTraining import NetworkFactoryModule
from PowerClassification.Utils.NetworkTraining import NetworkTrainingModule
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

EEG_channels = [  "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO7","PO3","PO4","PO8","OZ"]

Power_coefficients = ['Delta', 'Theta', 'Alpha', 'Beta']

if __name__ == '__main__':

    dataPath = r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\de-identified-pyprep-dataset-complete\02s\UI02"
    dataLoader = DataLoaderModule(dataFormat='freq')
    modelLoader = NetworkFactoryModule()
    trainModule = NetworkTrainingModule()

    validationKeys = ['5','6']

    # windowSize = 2
    totalSize = 100
    timesteps = int(totalSize/2)
    # features = 120

    #Load data
    # data = dataLoader.getDataSplitBySession(dataPath, timesteps=timesteps, powerBands=Power_coefficients)
    # data = dict(data)
    #
    # validationX = []
    # validationY = []
    # trainX = []
    # trainY = []
    # for key,item in data.items():
    #     if key in validatationKeys:
    #         validationX.append(item['X'])
    #         validationY.append(item['y'])
    #     else:
    #         trainX.append(item['X'])
    #         trainY.append(item['y'])
    #
    #
    # validationX = np.concatenate(validationX)
    # validationY = np.concatenate(validationY)
    # trainX = np.concatenate(trainX)
    # trainY = np.concatenate(trainY)

    #Load test data
    testPath = r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\de-identified-pyprep-dataset-bleeding\02s"
    test = dataLoader.getDataSplitBySession(testPath, timesteps=timesteps, powerBands=Power_coefficients)
    test = dict(test)
    testX = test['1']['X']
    testY = test['1']['y']

    #Change labels to categorical
    # validationY = to_categorical(validationY)
    # trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    #Load model
    model = load_model('model.h5')
    # model, modelName = modelLoader.createAdvanceLstmModel(timesteps, features)

    #Test model
    loss,acc = model.evaluate(testX, testY)
    y_predicted = model.predict(testX)

    print(model.metrics_names)
    print(loss, acc)

    #plot results
    fig, axes = plt.subplots(3,1)

    axes[0].stem(np.argmax(testY,axis=1), use_line_collection=True)
    # axes[1].stem(y_predicted[:,0], use_line_collection=True)
    # axes[2].stem(y_predicted[:,1], use_line_collection=True)

    x = np.arange(y_predicted.shape[0])
    colors = np.where(y_predicted[:, 0] < 0.5, 'y', 'k')
    axes[1].scatter(x,y_predicted[:, 0], c=colors)
    colors = np.where(y_predicted[:, 1] > 0.5, 'y', 'k')
    axes[2].scatter(x,y_predicted[:, 1], c=colors)
    plt.show()