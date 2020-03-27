'''
In the following script we are going to test if models trained with only the first three sessions
of each user  will improve to accuracy results

Experimental steps:
1)Select a testing user
2)Select a testing session
3)Select randomly a session for train and the remaining one to validation
4)Choose 1 session randomly from the other users for validation
5)The rest of the data goes to training
'''
import copy
import random

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
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch, validation_data=(X_val, y_val),
                        callbacks=callbacks, verbose=0)

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


def getDataFromEverybody():

    result = {}
    for u1 in allUsers:
        path = './data/users/{:}/'.format(u1)
        result[u1] = lstmClf.getDataSplitBySession(path)
    return result

if __name__ == '__main__':

    allUsers = ['jhony','jackie','ryan','juan']
    resultsContainer2 = []

    #Create dictionary of all users
    dataOfAllUsers = getDataFromEverybody()

    for user in [ 'jhony','jackie','ryan','juan']:

        theRestOfUsers = copy.copy(allUsers)
        theRestOfUsers.remove(user)

        resultsPath = './results/results_transfer5/'
        resultsFileName = resultsPath+'{:}_test.csv'.format(user)
        resultsContainer = []
        plotImageRootName = resultsPath + '{:}_test_'.format(user)

        testDataContainer = dataOfAllUsers[user]

        try:
            allSessions = ['1','2','3']

            # Iterate over all the available sessions creating different testing sets.
            for testingKey in ['1','2','3']:

                trainX, trainY = [], []
                valX, valY = [],[]
                testX, testY = [],[]

                #Select validation and training session
                sessions = copy.copy(allSessions)
                sessions.remove(testingKey)
                validationKey = random.sample(sessions, 1)[0]
                sessions.remove(validationKey)
                trainingKey = sessions[0]


                #Add testing user data
                trainX.append(testDataContainer[trainingKey]['X'])
                trainY.append(testDataContainer[trainingKey]['y'])
                valX.append(testDataContainer[validationKey]['X'])
                valY.append(testDataContainer[validationKey]['y'])
                testX.append(testDataContainer[testingKey]['X'])
                testY.append(testDataContainer[testingKey]['y'])

                #Add other users data
                for other in theRestOfUsers:
                    sess = ['1','2','3']
                    sess = random.sample(sess,len(sess))
                    trainDataContainer = dataOfAllUsers[other]

                    #Append first two sessions to training and the remaining one to validation
                    trainX.extend([trainDataContainer[k]['X'] for k in sess[:-1]])
                    trainY.extend([trainDataContainer[k]['y'] for k in sess[:-1]])
                    valX.append(trainDataContainer[sess[-1]]['X'])
                    valY.append(trainDataContainer[sess[-1]]['y'])

                    print('{} '.format(other),sess)


                trainX = np.concatenate(trainX)
                trainY = np.concatenate(trainY)
                valX = np.concatenate(valX)
                valY = np.concatenate(valY)
                testX = np.concatenate(testX)
                testY = np.concatenate(testY)


                # Convert labels to one-hot encoding
                trainY = to_categorical(trainY)
                valY = to_categorical(valY)
                testY = to_categorical(testY)

                #Normalize data
                X_mean = np.mean(trainX, axis=(0, 1))
                X_std = np.std(trainX, axis=(0, 1))

                trainX = (trainX - X_mean) / (X_std + 1e-18)
                valX = (valX - X_mean)/ (X_std + 1e-18)
                testX = (testX - X_mean) / (X_std + 1e-18)

                # Train Model
                #model = lstmClf.lstm2(*(trainX.shape[1], trainX.shape[2]))
                model = lstmClf.createAdvanceLstmModel(*(trainX.shape[1], trainX.shape[2]))

                history, model, earlyStopping = trainTestModel(model, trainX, trainY,valX,valY, testX, testY)
                evalTest0 = model.evaluate(testX, testY)
                K.clear_session()


                # Plot results
                fig, axes = plt.subplots(2, 1, sharex=True)
                axes[0].set_title('{:}_test_{:}_before'.format(user, testingKey))
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
                plt.savefig(plotImageRootName + '{:}_a_before.png'.format(testingKey))
                plt.close(fig)


                K.clear_session()

                # Add results
                data = {testingKey: [user,testingKey,validationKey, trainingKey, earlyStopping.epochOfMaxValidation,
                                     max(history.history['val_acc']), evalTest0[1]]}

                df1 = pd.DataFrame.from_dict(data, orient='index',
                                             columns=['user','test_session','validation_session',' training_session',
                                                      'epoch of max','validationAcc', 'testAcc'])
                resultsContainer.append(df1)
                resultsContainer2.append(df1)


        finally:
            # Create final Dataframe
            finalDf = pd.concat(resultsContainer)

            # Save results Dataframe to file
            try:
                finalDf.to_csv(resultsFileName, index=False)
            except:
                finalDf.to_csv(resultsPath + 'temp_{:}.csv'.format(user), index=False)

            # print("Finally Clause")

    finalDf = pd.concat(resultsContainer2)
    finalDf.to_csv(resultsPath + 'complete.csv', index=False)