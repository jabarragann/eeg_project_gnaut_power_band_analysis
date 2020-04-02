from typing import Any

import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax,LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
import re


TIMESTEPS = 12
FEATURES = 150



POWER_COEFFICIENTS = ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
#POWER_COEFFICIENTS = ['Theta','Alpha','Beta']
EEG_channels = [
        "FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
        "PZ", "P4", "P8", "PO3", "PO4", "OZ"]



def getData(train=True):
    # Create data column names
    columnNames = [x + '-' + y for x, y in product(EEG_channels, POWER_COEFFICIENTS)]

    #Set path
    if train:
        datapath = './data/train/'
    else:
        datapath = './data/test/'

    files = os.listdir(datapath)

    # Read all data files and concatenate them into a single dataframe.
    all_X = []
    all_y = []
    for f in files:
        #Get data frame
        d1 = pd.read_csv(datapath + f, sep=',', index_col=0)
        #Get data and labels
        X, y = d1[columnNames], d1['Label']
        #Get Number of features
        features = X.shape[1]
        timesteps = TIMESTEPS
        #Set data in LSTM format
        new_X, new_y = series_to_supervised(X, y, n_in=(timesteps-1), n_out=1, dropnan=True)
        new_X = new_X.reshape((new_X.shape[0], timesteps, features))
        #Append trial to dataset
        all_X.append(new_X)
        all_y.append(new_y)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    return X, y





# def lstm3(timesteps,features):
#     networkInput = Input(shape=(timesteps, features))
#
#     dropout1 = Dropout(rate=0.2)(networkInput)
#     hidden1 = Dense(30, activation='relu')(dropout1)
#     batchNorm1 = BatchNormalization()(hidden1)
#
#     hidden2 = LSTM(20, stateful=False, return_sequences=True)(batchNorm1)
#     dropout2 = Dropout(0.3)(hidden2)
#     hidden3 = LSTM(10, stateful=False)(dropout2)
#     dropout3= Dropout(0.3)(hidden3)
#     hidden4 = Dense(2, activation='linear')(dropout3)
#     networkOutput = Softmax()(hidden4)
#
#     model3 = Model(inputs=networkInput, outputs=networkOutput)
#     model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#
#     return model3

if __name__ == '__main__':

    getDataSplitBySession('./data/users/jackie/')
    random_seed = 47
    #Use trials 1-4 as training and trials 5-6 as testing
    # X, Y = getData(train=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state = random_seed)
    X_train, y_train = getData(train=True)
    X_test, y_test = getData(train=False)

    #Normalize data
    X_mean = np.mean(X_train, axis=(0,1))
    X_train = (X_train - X_mean)
    X_test = (X_test - X_mean)

    #Print data shape
    print("Train Shape", X_train.shape)
    print("Test Shape",  X_test.shape)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Train model
    batch = 256
    epochs = 400
    model = lstm2(*(TIMESTEPS, FEATURES))
    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test))

    # Print Max accuracy
    print("Training max accuracy: {:0.6f}".format(max(history.history['acc'])))
    print("Testing max accuracy:  {:0.6f}".format(max(history.history['val_acc'])))

    # Plot accuracy
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(history.history['acc'], label='train acc')
    axes[0].plot(history.history['val_acc'], label='test acc')
    axes[0].set_ylim([0.5,1])
    axes[1].plot(history.history['loss'], label='train loss')
    axes[1].plot(history.history['val_loss'], label='test loss')
    axes[0].legend()
    axes[1].legend()
    plt.show()


