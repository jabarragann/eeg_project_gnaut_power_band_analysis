import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from itertools import product
import re
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax,LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from twilio.rest import Client
import os
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras import backend as K
from pathlib import Path
import traceback
import copy

class Utils:
    @staticmethod
    def makeDir(path):
        try:
            os.mkdir(path)
        except Exception as e:
            print(e)
    @staticmethod
    def getDirectories(path):
        return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    @staticmethod
    def sendSMS(body, debug=False):
        try:
            account_sid = os.environ['TWILIO_ACCOUNT_SID']
            auth_token = os.environ['TWILIO_AUTH_TOKEN']

            client = Client(account_sid, auth_token)

            message = client.messages \
                .create(
                body=body,
                messaging_service_sid=os.environ['TWILIO_MESSAGING'],
                to='+13176031817'
            )
            if debug:
                print(message.sid)
        except Exception as e:
            print("Messaging service failed")
            print(e)

class DataLoaderModule:
    '''
        Utility to train data.
    '''
    # Channels PO7 and PO8 are not included
    EEG_CHANNELS = [
        "FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
        "PZ", "P4", "P8", "PO3", "PO4", "OZ"]
    POWER_COEFFICIENTS = ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    def series_to_supervised(self, data, labels, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        labels = labels[n_in:]
        return agg.values, labels.values

    def getDataSplitBySession(self, datapath, timesteps=None, powerBands = None, eegChannels = None):
        """
        Get a dictionary of all the different data sessions found in the datapath. A session is defined as all the data
        that was collected before the person takes of the sensor.

        :param datapath:
        :param timesteps:
        :param powerBands:
        :param eegChannels:
        :return:
        """
        if powerBands is None:
            powerBands = self.POWER_COEFFICIENTS
        if eegChannels is None:
            eegChannels = self.EEG_CHANNELS

        # Create data column names
        columnNames = [x + '-' + y for x, y in product(eegChannels, powerBands)]

        # Get a list of all the files
        files = os.listdir(datapath)

        # Create container
        dataContainer = defaultdict(lambda: defaultdict(list))

        # Read all data files and concatenate them into a single dataframe.
        for f in files:
            if f != "empty.py":
                # Get Session
                sessId = re.findall("S[0-9]_T[0-9]", f)[0][-4]

                # Get data frame
                print(datapath / f)
                d1 = pd.read_csv(datapath / f, sep=',', index_col=0)
                # Get data and labels
                X, y = d1[columnNames], d1['Label']
                # Get Number of features
                features = X.shape[1]
                timesteps = timesteps
                # Set data in LSTM format
                new_X, new_y = self.series_to_supervised(X, y, n_in=(timesteps - 1), n_out=1, dropnan=True)
                new_X = new_X.reshape((new_X.shape[0], timesteps, features))
                # Append trial to dataset
                dataContainer[sessId]['X'].append(new_X)
                dataContainer[sessId]['y'].append(new_y)

        for key, value in dataContainer.items():
            dataContainer[key]['X'] = np.concatenate(dataContainer[key]['X'])
            dataContainer[key]['y'] = np.concatenate(dataContainer[key]['y'])

        return dataContainer

    def getDataSplitBySessionByTrial(self,datapath, timesteps=None, powerBands=None):
        if powerBands is None:
            powerBands = self.POWER_COEFFICIENTS

        # Create data column names
        columnNames = [x + '-' + y for x, y in product(self.EEG_CHANNELS, powerBands)]

        # Get a list of all the files
        files = os.listdir(datapath)

        # Create container
        dataContainer = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Read all data files and concatenate them into a single dataframe.
        for f in files:
            if f != "empty.py":
                # Get Session and trial
                sessId = re.findall("S[0-9]_T[0-9]", f)[0][-4]
                trial = re.findall("S[0-9]_T[0-9]", f)[0][-1]

                # Get data frame
                d1 = pd.read_csv(datapath + f, sep=',', index_col=0)
                # Get data and labels
                X, y = d1[columnNames], d1['Label']
                # Get Number of features
                features = X.shape[1]
                timesteps = timesteps
                # Set data in LSTM format
                new_X, new_y = self.series_to_supervised(X, y, n_in=(timesteps - 1), n_out=1, dropnan=True)
                new_X = new_X.reshape((new_X.shape[0], timesteps, features))
                # Append trial to dataset
                dataContainer[sessId][trial]['X'] = new_X
                dataContainer[sessId][trial]['y'] = new_y

        return dataContainer

class NetworkFactoryModule:
    @staticmethod
    def lstm2(timesteps, features):
        networkInput = Input(shape=(timesteps, features))

        dropout1 = Dropout(rate=0.5)(networkInput)
        hidden1 = Dense(4, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.5)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        hidden2 = Bidirectional(LSTM(4, stateful=False, dropout=0.5))(batchNorm1)
        hidden3 = Dense(2, activation='linear')(hidden2)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1, 'simpleModel'

    @staticmethod
    def createAdvanceLstmModel(timesteps, features, isBidirectional=True, inputLayerNeurons=64, inputLayerDropout=0.3,
                               lstmSize=4):
        timesteps = timesteps
        features = features

        # Input layer
        networkInput = Input(shape=(timesteps, features))
        dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

        # First Hidden layer
        hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.5)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        # Choose if the network should be bidirectional
        if isBidirectional:
            lstmLayer = LSTM(lstmSize, stateful=False,
                             dropout=0.5, kernel_regularizer=regularizers.l2(0.05))
            # hidden2 = Bidirectional( LSTM(lstmSize, stateful=False, dropout=0.5), merge_mode='concat' ) (batchNorm1)
            hidden2 = Bidirectional(lstmLayer, merge_mode='concat')(batchNorm1)
        else:
            hidden2 = LSTM(lstmSize, stateful=False, dropout=0.5)(batchNorm1)

        hidden3 = Dense(2, activation='linear')(hidden2)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1, 'advanceModel'

class NetworkTrainingModule:

    '''
        Utility class to train networks
    '''

    def trainModelEarlyStop(self, model, X_train, y_train,
                            X_val, y_val, X_test, y_test,
                            batchSize=128, epochs=300, debug=True, verbose=1):
        """
            The following function trains a model with the given train data and validation data.
            At the same time it checks the performance of the model in the testing set through the
            EarlyStoppingCallback class. After training, the model is returned to the step in time
            where the best validation accuracy is obtained.

            :param model: Keras LSTM model to test
            :param X_train:
            :param y_train:
            :param X_val:
            :param y_val:
            :param X_test:
            :param y_test:
            :param batchSize: 512
            :param epochs: 300
            :param debug: True
            :param verbose:

            :return: kerasHistory, model, earlyStoppingInstance
        """
        # Normalize data
        # X_mean = np.mean(X_train, axis=(0,1))
        # X_std = np.std(X_train, axis = (0,1))

        # # Change data to float32
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        # Print data shape
        if debug:
            print("Train Shape", X_train.shape)
            print("Test Shape", X_test.shape)

        # Training parameters
        earlyStopCallback = self.EarlyStoppingCallback(2, additionalValSet=(X_test, y_test))

        #Train model
        K.clear_session()
        callbacks = [earlyStopCallback]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize,
                            validation_data=(X_val, y_val), callbacks=callbacks, verbose=verbose)

        #Get best weights
        model.set_weights(earlyStopCallback.bestWeights)

        return history, model, earlyStopCallback

    @staticmethod
    def createPlot(trainingHistory, title, path, show_plot=False, earlyStopCallBack=None):
        '''
        Create plots from training history
        :param trainingHistory:
        :param title:
        :param path:
        :param show_plot: Default=False
        :param earlyStopCallBack: Default = None
        :return:
        '''

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

    # Auxiliary classes
    class EarlyStoppingCallback(keras.callbacks.Callback):
        def __init__(self, modelNumber, additionalValSet=None):
            self.patience = 100
            self.stoppingCounter = 0
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
            self.stoppingCounter += 1

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
                self.stoppingCounter = 0

            # Test on additional set if there is one
            if self.testingOnAdditionalSet:
                evaluation = self.model.evaluate(self.val2X, self.val2Y, verbose=0)
                self.validationLoss2.append(evaluation[0])
                self.validationAcc2.append(evaluation[1])

            #Stop training if there hasn't been any improvement in 'Patience' epochs
            if self.stoppingCounter >= self.patience:
                self.model.stop_training = True

class CrossValidationRoutines:

    @staticmethod
    def userCrossValidation(lstmSteps, userDataPath, plotPath, user):
        """
        userCrossValidation will train a model on different fold of a single user. The method uses the following
        logic for the cross-validation.

        Step 1) Select a test session.
        Step 2) Randomly choose a validation session from the remaining sessions.
        Step 3) All the remaining session are the training set.
        Step 4) Repeat 1-3 for every session.

        :param lstmSteps:
        :param userDataPath:
        :param plotPath:
        :param user:
        :return:
        """

        dataLoaderModule  = DataLoaderModule()
        trainingModule = NetworkTrainingModule()
        factoryModule = NetworkFactoryModule()
        resultsContainer = []

        dataContainer = dataLoaderModule.getDataSplitBySession(userDataPath, timesteps=lstmSteps)

        try:
            # Iterate over all the available sessions creating different testing sets.
            for testingKey in dataContainer.keys():
                    print("Testing on session {:}".format(testingKey) )
                    testX = dataContainer[testingKey]['X']
                    testY = dataContainer[testingKey]['y']

                    # Select randomly one session for validation
                    availableSessions = list(dataContainer.keys())
                    availableSessions.remove(testingKey)
                    validationSession = random.choice(availableSessions)
                    availableSessions.remove(validationSession)

                    #Load validation data
                    valX = dataContainer[validationSession]['X']
                    valY = dataContainer[validationSession]['y']

                    # Use the rest of the sessions for the first round of training.
                    trainX, trainY = [], []
                    for trainingKey in availableSessions:
                        trainX.append(dataContainer[trainingKey]['X'])
                        trainY.append(dataContainer[trainingKey]['y'])

                    trainX = np.concatenate(trainX)
                    trainY = np.concatenate(trainY)


                    # Convert labels to one-hot encoding
                    trainY = to_categorical(trainY)
                    valY = to_categorical(valY)
                    testY = to_categorical(testY)

                    # Normalize data
                    X_mean = np.mean(trainX, axis=(0, 1))
                    X_std = np.std(trainX, axis=(0, 1))

                    trainX = (trainX - X_mean) / (X_std + 1e-18)
                    valX = (valX - X_mean) / (X_std + 1e-18)
                    testX = (testX - X_mean) / (X_std + 1e-18)

                    # Train Model
                    model, modelName = factoryModule.createAdvanceLstmModel(lstmSteps, trainX.shape[2])
                    history, model, earlyStopping = trainingModule.trainModelEarlyStop(model, trainX, trainY,
                                                                                       valX, valY, testX, testY, epochs=400,verbose=0)

                    #Create Plot
                    plotTitle = '{:}_test{:}_validation{:}'.format(user,testingKey, validationSession)
                    completePlotPath = plotPath / plotTitle
                    trainingModule.createPlot(history,plotTitle, completePlotPath,earlyStopCallBack=earlyStopping)
                    evalTrain = model.evaluate(trainX, trainY)[1]
                    evalValidation = model.evaluate(valX,valY)[1]
                    evalTest = model.evaluate(testX, testY)[1]
                    K.clear_session()


                    data = {testingKey: [user, modelName,testingKey,validationSession, evalTrain,evalValidation,evalTest, ]}

                    df1 = pd.DataFrame.from_dict(data, orient='index',
                                                 columns=[ 'User', 'ModelName','TestSession', 'ValidationSession', 'TrainAcc',
                                                           'ValidationAcc', 'TestAcc',])

                    resultsContainer.append(df1)
        except Exception as e:
            print("ERROR!!!!!!!!!!!!!!!!!!!")
            print(e)
            traceback.print_exc()

        finally:
            return pd.concat(resultsContainer)

    @staticmethod
    def userCrossValidationMultiUser(lstmSteps, dataPath, plotPath, testUser, listOfUsers, eegChannels=None):
        """
         userCrossValidationMultiUser will perform the crossvalidation process of a model. The main difference
        between userCrossValidation is that training data from multiple user will be added to the training and
        validation sets with the following procedure.

        1)Only data from one user will be in the testing set.
        2)The testing set will be a single session from the testing user.
        3)Another session of the testing user is added to the validation set.
        4)The rest of the sessions of the testing user go to the training set.

        Training and validation have data from multiple users.
        :param eegChannels: This parameter could be either None to use all the available EEG channels
         or a list of the channels that should be included in the data.
        :param lstmSteps:
        :param dataPath:
        :param plotPath:
        :param testUser:
        :param listOfUsers:
        :return:
        """

        dataLoaderModule = DataLoaderModule()
        trainingModule = NetworkTrainingModule()
        factoryModule = NetworkFactoryModule()
        resultsContainer = []

        testUserContainer = dataLoaderModule.getDataSplitBySession(dataPath / testUser, timesteps=lstmSteps,eegChannels=eegChannels)

        # Initially remove testing user from training and create train set without him
        trainingUsers = copy.copy(listOfUsers)
        trainingUsers.remove(testUser)

        firstTrainX = []
        firstTrainY = []
        firstValX = []
        firstValY = []
        logger = [[], [], []]
        for u in trainingUsers:
            # Open user data
            trainDataContainer = dataLoaderModule.getDataSplitBySession(dataPath / u, timesteps=lstmSteps, eegChannels= eegChannels )

            # Choose randomly 4 sessions for testing and 1 for validation
            availableSessions = np.array(list(trainDataContainer.keys()))
            idx = np.arange(len(availableSessions))
            idx = np.random.choice(idx, 4, replace=False)
            trSess = availableSessions[idx[:-1]]
            vlSess = availableSessions[idx[-1]]

            # create sets
            firstTrainX.append(np.concatenate([trainDataContainer[i]['X'] for i in trSess]))
            firstTrainY.append(np.concatenate([trainDataContainer[i]['y'] for i in trSess]))
            firstValX.append(np.concatenate([trainDataContainer[i]['X'] for i in vlSess]))
            firstValY.append(np.concatenate([trainDataContainer[i]['y'] for i in vlSess]))

            logger[0].append(u), logger[1].append(trSess), logger[2].append(vlSess)

        try:
            # Iterate over all the sessions of the testing user
            # Use one session for the test set and one for validation and the rest for training

            # Transform default dict to dict to avoid modifications of the container
            testUserContainer = dict(testUserContainer)

            for testingKey in testUserContainer.keys():

                # copy data from the other users
                trainX = copy.copy(firstTrainX)
                trainY = copy.copy(firstTrainY)
                valX = copy.copy(firstValX)
                valY = copy.copy(firstValY)

                # Select randomly one session for validation and use the rest for training
                availableSessions = list(testUserContainer.keys())
                availableSessions.remove(testingKey)
                validationSession = random.choice(availableSessions)
                availableSessions.remove(validationSession)


                # Add randomly sampled session to the validation set
                valX.append(testUserContainer[validationSession]['X'])
                valY.append(testUserContainer[validationSession]['y'])

                # Add the remaining sessions to the training set
                for j in availableSessions:
                    trainX.append(testUserContainer[j]['X'])
                    trainY.append(testUserContainer[j]['y'])

                # Concatenate all sessions for validation and training
                trainX = np.concatenate(trainX)
                trainY = np.concatenate(trainY)
                valX = np.concatenate(valX)
                valY = np.concatenate(valY)

                testX = testUserContainer[testingKey]['X']
                testY = testUserContainer[testingKey]['y']

                # Convert labels to one-hot encoding
                trainY = to_categorical(trainY)
                testY = to_categorical(testY)
                valY = to_categorical(valY)

                # Normalize data - with training set parameters
                globalMean = trainX.mean(axis=(0, 1))
                globalStd = trainX.std(axis=(0, 1))

                trainX = (trainX - globalMean) / (globalStd + 1e-18)
                valX = (valX - globalMean) / (globalStd + 1e-18)
                testX = (testX - globalMean) / (globalStd + 1e-18)

                # Train Model - First round
                model,modelName = factoryModule.createAdvanceLstmModel(*(trainX.shape[1], trainX.shape[2]))
                history, model, earlyStopping = trainingModule.trainModelEarlyStop(model, trainX, trainY, valX, valY, testX, testY,
                                                                                   epochs=700, verbose=0)

                evalTrain = model.evaluate(trainX, trainY,verbose=0)[1]
                evalValidation = model.evaluate(valX, valY, verbose=0)[1]
                evalTest = model.evaluate(testX, testY,verbose=1)[1]
                K.clear_session()


                # Plot results
                plotTitle = '{:}_test{:}_validation{:}'.format(testUser, testingKey, validationSession)
                completePlotPath = plotPath / plotTitle
                trainingModule.createPlot(history,plotTitle,completePlotPath,earlyStopCallBack=earlyStopping)


                # Save all the results
                data = {testingKey: [testUser, modelName,
                                     testingKey, validationSession,
                                     evalTrain, evalValidation, evalTest, ]}

                df1 = pd.DataFrame.from_dict(data, orient='index',
                                             columns=['User', 'ModelName', 'TestSession', 'ValidationSession', 'TrainAcc',
                                                      'ValidationAcc', 'TestAcc', ])

                resultsContainer.append(df1)

        except Exception as e:
            print("ERROR!!!!!!!!!!!!!!!!!!!")
            print(e)
            traceback.print_exc()

        finally:
            return pd.concat(resultsContainer)