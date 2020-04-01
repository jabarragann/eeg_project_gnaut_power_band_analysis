import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax,LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
import re


def lstmV1(timesteps,features):
    networkInput = Input(shape=(timesteps, features))

    dropout1 = Dropout(rate=0.5)(networkInput)
    hidden1 = Dense(4, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    hidden2 = LSTM(4, stateful=False, dropout=0.5)(batchNorm1)
    hidden3 = Dense(2, activation='linear')(hidden2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1

def lstmBidirectionalV1(timesteps,features):
    networkInput = Input(shape=(timesteps, features))

    dropout1 = Dropout(rate=0.5)(networkInput)
    hidden1 = Dense(8, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    hidden2 = Bidirectional( LSTM(10, stateful=False, dropout=0.5), merge_mode='concat' ) (batchNorm1)
    hidden3 = Dense(2, activation='linear')(hidden2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1

def lstmBidirectionalManyV1(timesteps,features):
    networkInput = Input(shape=(timesteps, features))
    dropout1 = Dropout(rate=0.5)(networkInput)

    hidden1 = Dense(8, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    hidden2 = Bidirectional( LSTM(10, stateful=False, dropout=0.5, return_sequences=True), merge_mode='concat' ) (batchNorm1)
    #hidden2 = LSTM(10, stateful=False, dropout=0.5, return_sequences=True)(batchNorm1)
    hidden3 = Dense(2, activation='linear')(hidden2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1

if __name__ == '__main__':

    bidi = lstmBidirectionalV1(15, 70)
    bidi.summary()

    bidiMany = lstmBidirectionalManyV1(15,70)
    bidiMany.summary()
