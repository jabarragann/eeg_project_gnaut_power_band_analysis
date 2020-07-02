import pandas as pd
import numpy as np
from PowerClassification.Utils.NetworkTraining import NetworkFactoryModule
from PowerClassification.Utils.NetworkTraining import NetworkTrainingModule
from PowerClassification.Utils.NetworkTraining import DataLoaderModule
from PowerClassification.HyperparameterTunning.HyperOptimShare import CommonFunctions as cf
from itertools import product



#Hyperparameters
lstmLayers = [2,3]
isBidirectional = [True, False]
lstmOutputCells = [4, 8]
windowSize = [10,20,30]

combinations = product(lstmLayers,isBidirectional,lstmOutputCells,windowSize)

#feature combinations
featureCombinations = pd.DataFrame(data =np.array(list(combinations)), columns = ["lstmLayers","isBidirectional","lstmOutputSize", "WindowSize"])
featureCombinations['sampleSize'] = 140
featureCombinations['timesteps'] = featureCombinations['sampleSize'] / featureCombinations['WindowSize']
featureCombinations['timesteps'] = featureCombinations['timesteps'].astype(int)
featureCombinations['features'] = 128
featureCombinations['mean'] = 0
featureCombinations['std'] = 0
featureCombinations['m0'],featureCombinations['m1'],featureCombinations['m2'] = 0,0,0
#set data type of columns to float64
featureCombinations = featureCombinations.astype(float)


#Create factory module
factory = NetworkFactoryModule()
trainer = NetworkTrainingModule()
loader = DataLoaderModule()

#Load data
data10 =  loader.getDataSplitBySession(cf.DATA_DICT['data_10'],timesteps=int(140/10),powerBands=cf.POWER_COEFFICIENTS,eegChannels=cf.EEG_CHANNELS)
data10 =  cf.separateIntoTrainValTest(data10)
data20 =  loader.getDataSplitBySession(cf.DATA_DICT['data_20'],timesteps=int(140/20),powerBands=cf.POWER_COEFFICIENTS,eegChannels=cf.EEG_CHANNELS)
data20 =  cf.separateIntoTrainValTest(data20)
data30 =  loader.getDataSplitBySession(cf.DATA_DICT['data_30'],timesteps=int(140/30),powerBands=cf.POWER_COEFFICIENTS,eegChannels=cf.EEG_CHANNELS)
data30 =  cf.separateIntoTrainValTest(data30)

#Train models with different combinations
for idx in range(featureCombinations.shape[0]):

    windowSize = featureCombinations.at[idx, 'WindowSize']

    ##Get data
    if windowSize == 10:
        trainX,trainY,valX,valY,testX,testY = data10
    elif windowSize == 20:
        trainX, trainY, valX, valY, testX, testY = data20
    elif windowSize == 30:
        trainX, trainY, valX, valY, testX, testY = data30
    else:
        raise Exception

    mean = np.zeros(3)
    for i in range(3):
        model = factory.hyperparameterTunning(featureCombinations.at[idx,'timesteps'],
                                              featureCombinations.at[idx,'features'],
                                              featureCombinations.at[idx,'lstmLayers'],
                                              featureCombinations.at[idx,'lstmOutputSize'],
                                              featureCombinations.at[idx,'isBidirectional'],)

        history, model, earlyStopCallback = trainer.trainModelEarlyStop(model,trainX,trainY,valX,valY,verbose=0,epochs=300)

        mean[i] = earlyStopCallback.maxValidationAcc
        featureCombinations.at[idx, 'm{:d}'.format(i)] = earlyStopCallback.maxValidationAcc

        print("Model {:d}, n training {:d}, valAcc {:f}".format(idx,i,mean[i]))

    #Train model -

    #save mean and std of combination
    featureCombinations.at[idx, 'mean'] = mean.mean()
    featureCombinations.at[idx, 'std'] = mean.std()

    #Save results
    featureCombinations.to_csv('./results.csv', index=None)

featureCombinations.to_csv('./results.csv',index=None)
