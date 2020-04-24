import random
import traceback
from collections import defaultdict
import pandas as pd
import copy
import re
from pathlib import Path
import numpy as np
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow.keras as keras

class NetworkTrainingModule:
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

class NetworkFactoryModule:
    @staticmethod
    def createModel(inputShape):
        regularizer = regularizers.l1(l=0.5)
        inputFeatures = Input(shape=inputShape)
        dropout1 = Dropout(rate=0.00)(inputFeatures)
        hidden1 = Dense(35, activation='relu', kernel_regularizer=regularizer)(dropout1)
        batchNorm1 = BatchNormalization()(hidden1)

        dropout2 = Dropout(rate=0.0)(batchNorm1)
        hidden2 = Dense(25, activation='relu', kernel_regularizer=regularizer)(dropout2)
        batchNorm2 = BatchNormalization()(hidden2)

        dropout3 = Dropout(rate=0.0)(batchNorm2)
        hidden3 = Dense(15, activation='relu', kernel_regularizer=regularizer)(dropout3)
        batchNorm3 = BatchNormalization()(hidden3)

        dropout4 = Dropout(rate=0.0)(batchNorm3)
        hidden4 = Dense(2)(dropout4)

        output = Softmax()(hidden4)
        model = Model(inputs=inputFeatures, outputs=output)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model

class DataLoaderModule:
    # SHIMMER_FEATURES = ["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50", "hr_mad", "sd1", "sd2", "s"]
    SHIMMER_FEATURES = ["riseTimeStd","riseTimeMean","fallTimeStd","fallTimeMean","peaskStd","valleysStd"]
    @staticmethod
    def getDataSplitBySession(dataPath, features = None):

        if features is None:
            features = DataLoaderModule.SHIMMER_FEATURES
        else:
            features = features

        dataContainer = defaultdict(lambda: defaultdict(list))

        for file in dataPath.rglob('*.txt'):
            df = pd.read_csv(file, sep=',', )

            session = re.findall("(?<=_S)[0-9](?=_T[0-9])", file.name)[0]
            print(file.name, session)

            y = df['label'].values
            X = df[features].values
            dataContainer[session]['y'].append(y)
            dataContainer[session]['X'].append(X)

        for key, value in dataContainer.items():
            dataContainer[key]['X'] = np.concatenate(dataContainer[key]['X'])
            dataContainer[key]['y'] = np.concatenate(dataContainer[key]['y'])
            dataContainer[key] = dict(dataContainer[key])

        dataContainer = dict(dataContainer)
        return dataContainer

class CrossValidationRoutines:

    @staticmethod
    def userCrossValidationMultiUser(dataPath, plotPath, testUser, listOfUsers):

        networkFactory = NetworkFactoryModule()
        dataLoaderModule = DataLoaderModule()
        trainingModule = NetworkTrainingModule()
        resultsContainer = []

        testUserContainer = dataLoaderModule.getDataSplitBySession(dataPath / testUser)

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
            trainDataContainer = dataLoaderModule.getDataSplitBySession(dataPath / u )

            # Choose randomly 1 session for validation and the rest for training
            availableSessions = list(trainDataContainer.keys())
            vlSess = random.choice(availableSessions)
            availableSessions.remove(vlSess)
            trSess = availableSessions

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

                # Clip outliers
                per25 = np.percentile(trainX, 25, axis=0, keepdims=1)
                per75 = np.percentile(trainX, 75, axis=0, keepdims=1)
                iqr1 = per75 - per25
                maxValues = per75 + 1.5 * iqr1
                minValues = per25 - 1.5 * iqr1
                trainX = np.clip(trainX, minValues, maxValues)
                xTest = np.clip(testX, minValues, maxValues)

                # Normalize data - with training set parameters
                globalMean = trainX.mean(axis=0)
                globalStd = trainX.std(axis=0)

                trainX = (trainX - globalMean) / (globalStd + 1e-18)
                valX = (valX - globalMean) / (globalStd + 1e-18)
                testX = (testX - globalMean) / (globalStd + 1e-18)

                # Train Model - First round
                networkName = networkFactory.createModel.__name__
                network = networkFactory.createModel(trainX.shape[1])
                history, model, earlyStopping = trainingModule.trainModelEarlyStop(network, trainX, trainY, valX, valY, testX, testY,
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
                data = {testingKey: [testUser, networkName,
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


if __name__ =='__main__':

    dataLoader = DataLoaderModule()

    pData=  Path('.').resolve().parent / "data" / "ShimmerPreprocessed" / "60s" / "jackie"
    print("Data directory: \n",pData)
    dataLoader.getDataSplitBySession(pData)



