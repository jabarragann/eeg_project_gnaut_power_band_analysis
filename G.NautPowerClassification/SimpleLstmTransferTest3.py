'''
In the following script, I am going to train the models using data from a single user.

Constraints of the experiment:
1)Only one session of the user will be in the testing set.
2)A randomly choose session is used as a validation set
2)The rest of the sessions will be used for the training set.
3)Two trials of the testing session will be used as a transfer set.

The experiment is going to be repeated for every session of the testing user.

Conclusion:
'''

import SimpleLstmClassification as lstmClf
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pandas as pd
from itertools import product
import tensorflow.keras as keras

#Global variables

lowTrials  = ['1','3','5']
highTrials = ['2','4','6']


def trainTestModel(model, X_train, y_train, X_val,y_val, X_test, y_test, lr=None):
    # Normalize data
    X_mean = np.mean(X_train, axis=(0, 1))
    X_std = np.std(X_train, axis = (0,1))

    # Print data shape
    print("Train Shape", X_train.shape)
    print("Test Shape", X_test.shape)

    # Train model
    batch = 256*2
    epochs = 300

    if lr is not None:
        print("Change Learning rate to ", lr)
        optim = Adam(learning_rate = lr)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])

    earlyStopCallback = EarlyStoppingCallback(2, additionalValSet=(X_test, y_test))
    callbacks = [earlyStopCallback]
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val), callbacks=callbacks)

    #Get best weights
    model.set_weights(earlyStopCallback.bestWeights)

    return history, model, earlyStopCallback

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

if __name__ == '__main__':

    # user = 'jhony','jackie','ryan','juan'
    for user in [ 'jhony','jackie','ryan','juan']:
        resultsPath = './results/results_transfer3/'
        resultsFileName = resultsPath+'{:}_test.csv'.format(user)
        resultsContainer = []
        plotImageRootName = resultsPath + '{:}_test_'.format(user)

        path = './data/users/{:}/'.format(user)
        dataContainer = lstmClf.getDataSplitBySessionByTrial(path)

        try:
            # Iterate over all the available sessions creating different testing sets.
            for testingKey in dataContainer.keys():

                #Sample randomly some trials to build the transfer subset.
                arr = np.array(list(product(lowTrials, highTrials)))
                idx = np.arange(arr.shape[0])
                idx = np.random.choice(idx, 1, replace=False)

                for lt,ht in arr[idx]:
                    # Get testing trials and transfer trials
                    #Use 2 of the trials for the transfer set and the rest for the testing set.
                    testTrials = np.array(list(dataContainer[testingKey].keys()))
                    idx =  np.argwhere(testTrials == lt), np.argwhere(testTrials == ht)
                    testTrials = np.delete(testTrials, idx)
                    transferTrials = [lt, ht]

                    transferX = np.concatenate([dataContainer[testingKey][i]['X'] for i in transferTrials])
                    transferY = np.concatenate([dataContainer[testingKey][i]['y'] for i in transferTrials])
                    testX = np.concatenate([dataContainer[testingKey][i]['X'] for i in testTrials])
                    testY = np.concatenate([dataContainer[testingKey][i]['y'] for i in testTrials])

                    #Select randomly one session for validation
                    import random
                    availableSessions = list(dataContainer.keys())
                    availableSessions.remove(testingKey)
                    validationSession = random.choice(availableSessions)
                    availableSessions.remove(validationSession)

                    valX = np.concatenate([dataContainer[validationSession][i]['X'] for i in dataContainer[validationSession].keys()])
                    valY = np.concatenate([dataContainer[validationSession][i]['y'] for i in dataContainer[validationSession].keys()])

                    # Use the rest of the sessions for the first round of training.
                    trainX, trainY = [], []
                    for trainingKey in availableSessions:
                        for trialKey in dataContainer[trainingKey].keys():
                                trainX.append(dataContainer[trainingKey][trialKey]['X'])
                                trainY.append(dataContainer[trainingKey][trialKey]['y'])

                    trainX = np.concatenate(trainX)
                    trainY = np.concatenate(trainY)

                    #Augment transfer set by injecting random noise
                    augmentedTraining = transferX + 0.15 * np.random.randn(*transferX.shape)
                    transferX = np.concatenate((transferX, augmentedTraining))
                    transferY = np.concatenate((transferY, transferY))

                    # Convert labels to one-hot encoding
                    trainY = to_categorical(trainY)
                    valY = to_categorical(valY)
                    transferY = to_categorical(transferY)
                    testY = to_categorical(testY)

                    #Normalize data
                    X_mean = np.mean(trainX, axis=(0, 1))
                    X_std = np.std(trainX, axis=(0, 1))

                    trainX = (trainX - X_mean) / (X_std + 1e-18)
                    valX = (valX - X_mean)/ (X_std + 1e-18)
                    transferX = (transferX - X_mean) / (X_std + 1e-18)
                    testX = (testX - X_mean) / (X_std + 1e-18)

                    # Train Model
                    #model = lstmClf.lstm2(*(trainX.shape[1], trainX.shape[2]))
                    model = lstmClf.createAdvanceLstmModel(*(trainX.shape[1], trainX.shape[2]))

                    history, model, earlyStopping = trainTestModel(model, trainX, trainY,valX,valY, testX, testY)
                    evalTrain0 = model.evaluate(trainX, trainY)
                    evalTest0 = model.evaluate(testX, testY)
                    K.clear_session()


                    # Plot results
                    fig, axes = plt.subplots(2, 1, sharex=True)
                    axes[0].set_title('{:}_test_{:}_{:}_{:}_before'.format(user, testingKey,lt,ht))
                    axes[0].plot(history.history['acc'], label='train acc')
                    axes[0].plot(history.history['val_acc'], label='val acc')
                    axes[0].plot(earlyStopping.validationAcc2, label='test acc')
                    axes[0].axvline(earlyStopping.epochOfMaxValidation, 0, 1)
                    axes[0].set_ylim([0.5, 1])
                    axes[1].plot(history.history['loss'], label='train loss')
                    axes[1].plot(history.history['val_loss'], label='test loss')
                    axes[0].legend()
                    axes[1].legend()
                    # plt.show()

                    # Save Plot
                    plt.savefig(plotImageRootName + '{:}_{:}_{:}_a_before.png'.format(testingKey,lt,ht))
                    plt.close(fig)

                    #Combine the train and transfer sets

                    #Train another model
                    TRANSFER_EPOCHS = 300
                    MINIBATCH = 256
                    MODEL_NUMBER = 2

                    # Change model optimization algorithm
                    optim = keras.optimizers.SGD(learning_rate=0.015, momentum=0.02, nesterov=False)
                    lossFunc = keras.losses.categorical_crossentropy
                    model.compile(optim, loss='categorical_crossentropy', metrics=['acc'])

                    transferEarlyStopCallback = EarlyStoppingCallback(MODEL_NUMBER, additionalValSet=(testX, testY))
                    callbacks = [transferEarlyStopCallback]
                    evalTestBefore = model.evaluate(testX,testY)[1]
                    transferHistory = model.fit(transferX, transferY, validation_data=(testX, testY),
                                                 epochs=TRANSFER_EPOCHS, batch_size=MINIBATCH, verbose=1, callbacks=callbacks,
                                                 shuffle=True)

                    evalTestAfter = model.evaluate(testX, testY)[1]
                    K.clear_session()

                    # Plot results
                    fig, axes = plt.subplots(2, 1, sharex=True)
                    axes[0].set_title('{:}_test_{:}_{:}_{:}after'.format(user, testingKey,lt,ht))
                    axes[0].plot(transferHistory.history['acc'], label='train acc')
                    axes[0].plot(transferHistory.history['val_acc'], label='test acc')
                    axes[0].set_ylim([0.5, 1])
                    axes[1].plot(transferHistory.history['loss'], label='train loss')
                    axes[1].plot(transferHistory.history['val_loss'], label='test loss')
                    axes[0].legend()
                    axes[1].legend()

                    # plt.show()

                    # Save Plot
                    plt.savefig(plotImageRootName + '{:}_{:}_{:}_b_after.png'.format(testingKey,lt,ht))
                    plt.close(fig)

                    # Add results
                    difference = transferHistory.history['val_acc'][-1] - history.history['val_acc'][-1]
                    data = {testingKey: [testingKey,validationSession,
                                         lt,ht, max(history.history['val_acc']), evalTestBefore,
                                         transferHistory.history['val_acc'][-1], difference]}
                    df1 = pd.DataFrame.from_dict(data, orient='index',
                                                 columns=['test_session','validation_session',
                                                          'low_trial','high_trial','max validation','test_acc_before',
                                                          'test_acc_after','Difference'])
                    resultsContainer.append(df1)


        finally:
            # Create final Dataframe
            finalDf = pd.concat(resultsContainer)

            # Save results Dataframe to file
            try:
                finalDf.to_csv(resultsFileName, index=False)
            except:
                finalDf.to_csv(resultsPath + 'temp_{:}.csv'.format(user), index=False)

            # print("Finally Clause")
