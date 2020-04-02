import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import product
import re
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax,LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

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

    def getDataSplitBySession(self, datapath, timesteps=None, powerBands = None):

        if powerBands is None:
            powerBands = self.POWER_COEFFICIENTS

        # Create data column names
        columnNames = [x + '-' + y for x, y in product(self.EEG_CHANNELS, powerBands)]

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
                print(datapath + f)
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

        return model1

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

        return model1
class NetworkTrainingModule:

    '''
        Utility class to train networks
    '''

    def trainModelEarlyStop(self, model, X_train, y_train,
                            X_val, y_val, X_test, y_test,
                            batchSize=256*2, epochs=300, debug=True):
        '''
            The following function trains a model with the given train data and validation data.
            At the same time it checks the performance of the model in the testing set through the
            EarlyStoppingCallback class. After training, the model is returned to the step in time
            where the best validation accuracy is obtained.

            :param model:
            :param X_train:
            :param y_train:
            :param X_val:
            :param y_val:
            :param X_test:
            :param y_test:
            :param batchSize: 512
            :param epochs: 300
            :param debug: True

            :return: kerasHistory, model, earlyStoppingInstance
        '''

        # Normalize data
        # X_mean = np.mean(X_train, axis=(0,1))
        # X_std = np.std(X_train, axis = (0,1))

        # Print data shape
        if debug:
            print("Train Shape", X_train.shape)
            print("Test Shape", X_test.shape)

        # Training parameters
        batchSize = batchSize
        epochs = epochs
        earlyStopCallback = self.EarlyStoppingCallback(2, additionalValSet=(X_test, y_test))

        #Train model
        callbacks = [earlyStopCallback]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize,
                            validation_data=(X_val, y_val), callbacks=callbacks)

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

            # Test on additional set if there is one
            if self.testingOnAdditionalSet:
                evaluation = self.model.evaluate(self.val2X, self.val2Y, verbose=0)
                self.validationLoss2.append(evaluation[0])
                self.validationAcc2.append(evaluation[1])