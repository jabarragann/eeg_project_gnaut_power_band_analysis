import pandas as pd
from PowerClassification.HyperparameterTunning.HyperOptimShare import CommonFunctions as cf
from PowerClassification.Utils.NetworkTraining import NetworkFactoryModule
from PowerClassification.Utils.NetworkTraining import NetworkTrainingModule
from PowerClassification.Utils.NetworkTraining import DataLoaderModule
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create factory module
    factory = NetworkFactoryModule()
    trainer = NetworkTrainingModule()
    loader = DataLoaderModule()

    #Read data
    df = pd.read_csv("./results2.csv",sep=',')
    df = df.sort_values(by ='mean',ascending=False)

    #Get Best model
    bestModelParams  = df.iloc[0]

    print("Timesteps {}, features {}, lstmLayers {}, lstmOutput {}, isBidirectional {}, WindowSize {}".format(
        bestModelParams['timesteps'],
        bestModelParams['features'],
        bestModelParams['lstmLayers'],
        bestModelParams['lstmOutputSize'],
        bestModelParams['isBidirectional'],
        bestModelParams['WindowSize']))

    #Create model
    windowSize = bestModelParams['WindowSize']
    model = factory.experimentingWithLowerDropout(  bestModelParams['timesteps'],
                                                    bestModelParams['features'],
                                                    bestModelParams['lstmLayers'],
                                                    bestModelParams['lstmOutputSize'],
                                                    bestModelParams['isBidirectional'], )
    # model = factory.hyperparameterTunning(  bestModelParams['timesteps'],
    #                                         bestModelParams['features'],
    #                                         bestModelParams['lstmLayers'],
    #                                         bestModelParams['lstmOutputSize'],
    #                                         bestModelParams['isBidirectional'], )



    # Load data
    #Get data
    if windowSize == 10:
        data10 = loader.getDataSplitBySession(cf.DATA_DICT['data_10'], timesteps=int(140 / 10),
                                              powerBands=cf.POWER_COEFFICIENTS, eegChannels=cf.EEG_CHANNELS)
        data10 = cf.separateIntoTrainValTest(data10)
        trainX, trainY, valX, valY, testX, testY = data10
    elif windowSize == 20:
        data20 = loader.getDataSplitBySession(cf.DATA_DICT['data_20'], timesteps=int(140 / 20),
                                              powerBands=cf.POWER_COEFFICIENTS, eegChannels=cf.EEG_CHANNELS)
        data20 = cf.separateIntoTrainValTest(data20)
        trainX, trainY, valX, valY, testX, testY = data20
    elif windowSize == 30:
        data30 = loader.getDataSplitBySession(cf.DATA_DICT['data_30'], timesteps=int(140 / 30),
                                              powerBands=cf.POWER_COEFFICIENTS, eegChannels=cf.EEG_CHANNELS)
        data30 = cf.separateIntoTrainValTest(data30)
        trainX, trainY, valX, valY, testX, testY = data30
    else:
        raise Exception


    # Train model
    history, model, earlyStopCallback = trainer.trainModelEarlyStop(model, trainX, trainY, valX, valY,
                                                                    verbose=0, epochs=300)


    results = model.evaluate(testX,testY)

    y_score = model.predict(testX)
    print("Best model. loss: {:0.4f}, acc: {:0.4f}".format(results[0],results[1]))

    fpr, tpr, thresh = roc_curve(testY[:,1], y_score[:,1])
    roc_auc = auc(fpr, tpr)

    trainer.createPlot(history, "Training best model", "./bestTrainPlot", show_plot=False, earlyStopCallBack=earlyStopCallback, save=True)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig("AUC_ROC" + '.png')
    plt.show()




