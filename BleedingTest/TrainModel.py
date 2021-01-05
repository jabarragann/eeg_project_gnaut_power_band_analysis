
from PowerClassification.Utils.NetworkTraining import DataLoaderModule
from PowerClassification.Utils.NetworkTraining import NetworkFactoryModule
from PowerClassification.Utils.NetworkTraining import NetworkTrainingModule
from tensorflow.keras.utils import to_categorical
import numpy as np


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
    # validationKeys = ['2','4']

    windowSize = 2
    totalSize = 100
    timesteps = int(totalSize/2)
    features = 120

    #Load data
    data = dataLoader.getDataSplitBySession(dataPath, timesteps=timesteps, powerBands=Power_coefficients)
    data = dict(data)

    validationX = []
    validationY = []
    trainX = []
    trainY = []
    for key,item in data.items():
        if key in validationKeys:
            validationX.append(item['X'])
            validationY.append(item['y'])
        else:
            trainX.append(item['X'])
            trainY.append(item['y'])


    validationX = np.concatenate(validationX)
    validationY = np.concatenate(validationY)
    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)

    #Change labels to categorical
    validationY = to_categorical(validationY)
    trainY = to_categorical(trainY)

    #Load model
    model,modelName = modelLoader.createAdvanceLstmModel(timesteps,features)


    #Train model
    kerasHistory, model, earlyStoppingInstance=trainModule.trainModelEarlyStop(model,trainX,trainY,validationX,validationY)
    trainModule.createPlot(kerasHistory,"Single user model","./graph", earlyStopCallBack= earlyStoppingInstance,show_plot=True)


    #save model
    model.save('./model.h5')