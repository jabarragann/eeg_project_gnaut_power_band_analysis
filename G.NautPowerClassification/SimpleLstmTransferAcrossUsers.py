import SimpleLstmClassification as lstmClf
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pandas as pd
import copy
import tensorflow.keras as keras
from itertools import product

#Global variables

lowTrials  = ['1','3','5']
highTrials = ['2','4','6']

def normalizeAcrossTime (data):

    globalMean = data.mean(axis=(0,1))
    globalStd  = data.std (axis=(0,1))

    data = (data - globalMean) / (globalStd + 1e-18)

    return data

def trainTestModel(X_train, y_train, X_test, y_test, features=None, timesteps=None, lr=None):
    # # Normalize data
    # X_mean = np.mean(X_train, axis=(0, 1))
    # X_train = (X_train - X_mean)
    # X_test = (X_test - X_mean)

    # Print data shape
    print("Train Shape", X_train.shape)
    print("Test Shape", X_test.shape)

    # Train model
    batch = 256
    epochs = 300

    model = getNewModel(*(timesteps, features))

    if lr is not None:
        print("Change Learning rate to ", lr)
        optim = Adam(learning_rate = lr)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])


    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test))

    return history, model

#Create plot and save it
def createPlot(trainingHistory, title, path, show_plot=False, earlyStopCallBack=None):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_title(title)
    axes[0].plot(trainingHistory.history['acc'], label='train acc')
    axes[0].plot(trainingHistory.history['val_acc'], label='val acc')
    axes[0].set_ylim([0.5, 1])
    axes[1].plot(trainingHistory.history['loss'], label='train loss')
    axes[1].plot(trainingHistory.history['val_loss'], label='val loss')
    axes[0].legend()
    axes[1].legend()

    #Testing acc
    if earlyStopCallBack:
        axes[0].plot(earlyStopCallBack.validationAcc2, label='test acc')
        axes[0].axvline(x=earlyStopCallBack.epochOfMaxValidation)

    if show_plot:
        plt.show()

    plt.savefig(path)
    plt.close(fig)


#Auxiliary classes
class EarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, modelNumber, additionalValSet = None):
        self.maxValidationAcc = 0.0
        self.epochOfMaxValidation = -1
        self.epoch = 0
        self.modelNumber = modelNumber
        self.bestWeights = None

        self.trainingLosses = None
        self.trainingAcc = None
        self.validationLosses = None
        self.validationAcc = None

        self.testingOnAdditionalSet = False

        if additionalValSet is not None:
            self.testingOnAdditionalSet = True
            self.validationLoss2 = []
            self.validationAcc2 = []
            self.val2X = additionalValSet[0]
            self.val2Y = additionalValSet[1]

    def on_train_begin(self, logs):
        self.trainingLosses = []
        self.trainingAcc = []
        self.validationLosses = []
        self.validationAcc = []

    def on_epoch_end(self, batch, logs):
        self.epoch += 1

        # Save Training Information
        self.trainingLosses.append(logs.get('loss'))
        self.trainingAcc.append(logs.get('acc'))
        self.validationLosses.append(logs.get('val_loss'))
        self.validationAcc.append(logs.get('val_acc'))

        # Save Best Models
        if logs.get('val_acc') > self.maxValidationAcc:
            self.epochOfMaxValidation = self.epoch
            self.bestWeights = self.model.get_weights()
            self.maxValidationAcc = logs.get('val_acc')

        #Test on additional set if there is one
        if self.testingOnAdditionalSet:
            evaluation = self.model.evaluate(self.val2X, self.val2Y, verbose=0)
            self.validationLoss2.append(evaluation[0])
            self.validationAcc2.append(evaluation[1])

def getNewModel(a,b):
    return lstmClf.createAdvanceLstmModel(*(trainX.shape[1], trainX.shape[2]))
    #return lstmClf.lstm2(*(trainX.shape[1], trainX.shape[2]))

allUsers = ['ryan','jhony','jackie','juan']
if __name__ == '__main__':

    resultsContainer2 = []

    #Select 1 test user
    for testUser in ['jackie','juan','ryan','jhony']:
        resultsPath = './results/across-subject/'
        resultsFileName = resultsPath+'{:}_test.csv'.format(testUser)
        resultsContainer = []
        plotImageRootName = resultsPath + '{:}_test_'.format(testUser)

        path = './data/users/{:}/'.format(testUser)
        testUserContainer = lstmClf.getDataSplitBySession(path)

        #Build training and validation. The test user is not included in these sets.
        trainingUsers = copy.copy(allUsers)
        trainingUsers.remove(testUser)

        trainX = []
        trainY = []
        valX = []
        valY = []
        logger = [[],[],[]]
        for u in trainingUsers:
            p = './data/users/{:}/'.format(u)
            dataContainer = lstmClf.getDataSplitBySession(p)

            availableSessions = np.array(list(dataContainer.keys()))
            idx = np.arange(len(availableSessions))
            idx = np.random.choice(idx, 4, replace=False)
            trSess = availableSessions[idx[:-1]]
            vlSess = availableSessions[idx[ -1]]

            trainX.append(np.concatenate([dataContainer[i]['X'] for i in trSess]))
            trainY.append(np.concatenate([dataContainer[i]['y'] for i in trSess]))
            valX.append(np.concatenate([dataContainer[i]['X'] for i in vlSess]))
            valY.append(np.concatenate([dataContainer[i]['y'] for i in vlSess]))

            logger[0].append(u), logger[1].append(trSess), logger[2].append(vlSess)

        trainX = np.concatenate(trainX)
        trainY = np.concatenate(trainY)
        valX = np.concatenate(valX)
        valY = np.concatenate(valY)

        #Normalize data Across time
        # trainX = normalizeAcrossTime(trainX)
        # valX = normalizeAcrossTime(valX)
        globalMean = trainX.mean(axis=(0, 1))
        globalStd = trainX.std(axis=(0, 1))

        trainX = (trainX - globalMean) / (globalStd + 1e-18)
        valX = (valX - globalMean) / (globalStd + 1e-18)

        try:
                # Convert labels to one-hot encoding
                trainY = to_categorical(trainY)
                valY = to_categorical(valY)

                # Train Model
                history, model = trainTestModel(trainX, trainY, valX, valY, timesteps=trainX.shape[1], features=trainX.shape[2])

                #Save training plot
                plotTitle = '{:}_test_before'.format(testUser)
                plotPath = plotImageRootName + '{:}_test_before'.format(testUser)
                createPlot(history,plotTitle, plotPath)

                #Transfer learning
                #Save best Weights
                bestWeightsFirstRound = model.get_weights()

                ##Repeat this a few times

                for i in range(3):
                    #Pick two random sessions for testing and the rest for re training
                    allTesting = np.array(list(testUserContainer.keys()))
                    idx = np.arange(allTesting.shape[0])
                    idx = np.random.choice(idx, len(allTesting), replace=False)

                    #Transfer Val-test-Train
                    transferSets = [allTesting[idx[0]], allTesting[idx[1:3]], allTesting[idx[3:5]]]
                    # transferSets = ['8'	,['1','6'],['3','4']]

                    transferValX = testUserContainer[transferSets[0]]['X']
                    transferValY = testUserContainer[transferSets[0]]['y']
                    transferValY = to_categorical(transferValY)

                    testX = np.concatenate([testUserContainer[i]['X'] for i in transferSets[1]])
                    testY = np.concatenate([testUserContainer[i]['y'] for i in transferSets[1]])
                    testY = to_categorical(testY)

                    transferX = np.concatenate([testUserContainer[i]['X'] for i in transferSets[2]])
                    transferY = np.concatenate([testUserContainer[i]['y'] for i in transferSets[2]])
                    transferY = to_categorical(transferY)

                    #Normalize data across-time
                    # globalTransferMean = transferX.mean(axis=(0, 1))
                    # globalTransferStd = transferX.std(axis=(0, 1))

                    # transferX    = (transferX - globalTransferMean) / (globalTransferStd + 1e-18)
                    # transferValX = (transferValX - globalTransferMean) / (globalTransferStd + 1e-18)
                    # testX = (testX - globalTransferMean) / (globalTransferStd + 1e-18)

                    transferX = (transferX - globalMean) / (globalStd + 1e-18)
                    transferValX = (transferValX - globalMean) / (globalStd + 1e-18)
                    testX = (testX - globalMean) / (globalStd + 1e-18)
                    # transferX = normalizeAcrossTime(transferX)
                    # transferValX = normalizeAcrossTime(transferValX)
                    # testX = normalizeAcrossTime(testX)

                    #Re set model weights
                    model.set_weights(bestWeightsFirstRound)

                    #Get results before retraining
                    resultsBefore = model.evaluate(testX, testY)

                    #Re train model
                    valResults = model.evaluate(valX, valY)
                    TRANSFER_EPOCHS = 300
                    MINIBATCH =256
                    MODEL_NUMBER = 2


                    transferEarlyStopCallback = EarlyStoppingCallback(MODEL_NUMBER, additionalValSet=(testX, testY))
                    callbacks = [transferEarlyStopCallback]
                    transfer_history = model.fit(transferX, transferY, validation_data=(transferValX, transferValY),
                                                    callbacks=callbacks,
                                                    epochs=TRANSFER_EPOCHS, batch_size=MINIBATCH, verbose=1,
                                                    shuffle=True)
                    # history = model.fit(transferX, transferY, validation_data=(testX, testY),
                    #               epochs=TRANSFER_EPOCHS, batch_size=MINIBATCH, verbose=1, shuffle=True)

                    # Save training plot
                    plotTitle = '{:}_reTrain_{:d}'.format(testUser,i)
                    plotPath = plotImageRootName + '{:}_reTrain_{:d}.png'.format(testUser,i)
                    createPlot(transfer_history, plotTitle, plotPath, earlyStopCallBack = transferEarlyStopCallback)

                    #Set best Weights
                    model.set_weights(transferEarlyStopCallback.bestWeights)

                    # Get results before retraining
                    resultsAfter = model.evaluate(testX, testY)
                    maxTransferVal = transferEarlyStopCallback.maxValidationAcc

                    #Train model from scratch
                    scratchModel = getNewModel(*(trainX.shape[1], trainX.shape[2]))

                    scratchEarlyStopCallback = EarlyStoppingCallback(MODEL_NUMBER, additionalValSet=(testX, testY))
                    callbacks = [scratchEarlyStopCallback]
                    scratch_history = scratchModel.fit(transferX, transferY, validation_data=(transferValX, transferValY),
                                                 callbacks=callbacks,
                                                 epochs=TRANSFER_EPOCHS, batch_size=MINIBATCH, verbose=1,
                                                 shuffle=True)

                    #Create plot
                    plotTitle = '{:}_scratch_{:d}'.format(testUser, i)
                    plotPath = plotImageRootName + '{:}_scratch_{:d}.png'.format(testUser, i)
                    createPlot(scratch_history, plotTitle, plotPath,  earlyStopCallBack = scratchEarlyStopCallback)

                    # Set best Weights
                    model.set_weights(scratchEarlyStopCallback.bestWeights)
                    scratchResults = model.evaluate(testX, testY)

                    #Save all the results
                    data = {testUser: [i, testUser, logger[0], " ".join([str(t) for t in logger[1]])
                                        , logger[2],transferSets[2],transferSets[0],transferSets[1],
                                        valResults[1],maxTransferVal, resultsBefore[1],resultsAfter[1],
                                       scratchResults[1]]}

                    df1 = pd.DataFrame.from_dict(data, orient='index',
                                                 columns=['attempt','user','others','others_train','others_val',
                                                          'transferTrain','TransferVal','TransferTest','max_val','maxTransferVal',
                                                          'testBefore','testAfter','scratch'])
                    resultsContainer.append(df1)
                    resultsContainer2.append(df1)
                    valResults = 0

                # #Test Model
                # for key, dataDict in testUserContainer.items():
                #     testX = dataDict['X']
                #     testY = to_categorical(dataDict['y'])
                #
                #     results = model.evaluate(testX, testY)
                #
                #     data = {testUser: [1, testUser, logger[0], " ".join([str(t) for t in logger[1]])
                #                         , logger[2], history.history['val_acc'][-1], results[1]]}
                #
                #     df1 = pd.DataFrame.from_dict(data, orient='index',
                #                                  columns=['attempt','user','others','others_train','others_val','max_val','max_test'])
                #     resultsContainer.append(df1)
                #     resultsContainer2.append(df1)

                K.clear_session()

        finally:
            # Create final Dataframe
            finalDf = pd.concat(resultsContainer)

            # Save results Dataframe to file
            try:
                finalDf.to_csv(resultsFileName, index=False)
            except:
                finalDf.to_csv(resultsPath + 'temp_{:}.csv'.format(testUser), index=False)

            # print("Finally Clause")

    finalDf = pd.concat(resultsContainer2)
    finalDf.to_csv(resultsPath + 'complete.csv', index=False)