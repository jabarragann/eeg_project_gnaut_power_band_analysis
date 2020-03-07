from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Channels PO7 and PO8 are not included
EEG_channels = [
                    "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                    "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                    "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                    "PZ","P4","P8","PO3","PO4","OZ"]

Power_coefficients = ['Low','Delta','Theta','Alpha','Beta']

def createModel(inputShape):
    inputFeatures = Input(shape=inputShape)
    dropout1 = Dropout(rate=0.1)(inputFeatures)
    hidden1 = Dense(20, activation='relu')(dropout1)
    batchNorm1 = BatchNormalization()(hidden1)

    dropout2 = Dropout(rate=0.3)(batchNorm1)
    hidden2 = Dense(10, activation='relu')(dropout2)
    batchNorm2 = BatchNormalization()(hidden2)

    dropout3 = Dropout(rate=0.3)(batchNorm2)
    hidden3 = Dense(8)(dropout3)
    batchNorm3 = BatchNormalization()(hidden3)

    dropout4 = Dropout(rate=0.3)(batchNorm3)
    hidden4 = Dense(2)(dropout4)

    output = Softmax()(hidden4)
    model = Model(inputs=inputFeatures, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def createModel2(inputShape):
    inputFeatures = Input(shape=inputShape)
    dropout1 = Dropout(rate=0.1)(inputFeatures)
    hidden1 = Dense(40, activation='relu')(dropout1)
    batchNorm1 = BatchNormalization()(hidden1)

    dropout2 = Dropout(rate=0.3)(batchNorm1)
    hidden2 = Dense(20, activation='relu')(dropout2)
    batchNorm2 = BatchNormalization()(hidden2)

    dropout3 = Dropout(rate=0.3)(batchNorm2)
    hidden3 = Dense(16)(dropout3)
    batchNorm3 = BatchNormalization()(hidden3)

    dropout4 = Dropout(rate=0.3)(batchNorm3)
    hidden4 = Dense(2)(dropout4)

    output = Softmax()(hidden4)
    model = Model(inputs=inputFeatures, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def getData(train=True):
    # Create data column names
    columnNames = [x + '-' + y for x, y in product(EEG_channels, Power_coefficients)]

    #Set path
    if train:
        datapath = './data/train/'
    else:
        datapath = './data/test/'

    files = os.listdir(datapath)

    # Read all data files and concatenate them into a single dataframe.
    all = []
    for f in files:
        d1 = pd.read_csv(datapath + f, sep=',', index_col=0)
        all.append(d1)
    dataset = pd.concat(all)

    X, y = dataset[columnNames].values, dataset['Label'].values

    return X,y

if __name__ == '__main__':

    #Use trials 1-4 as training and trials 5-6 as testing
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=False)
    X_train, y_train = getData(train=True)
    X_test, y_test = getData(train=False)

    #Normalization of the features
    meanVector = np.mean(X_train, axis=0)
    X_train =  X_train - meanVector
    X_test = X_test - meanVector

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #PCA dimensionality reduction
    # pca = PCA(n_components=140)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test  = pca.transform(X_test)

    # clf = RandomForestClassifier(n_estimators=2)
    # clf = KNeighborsClassifier(n_neighbors=60)
    # clf = SVC(gamma='auto')

    # clf.fit(X_train, y_train)
    # y_hat = clf.predict(X_train)
    # trainAcc = accuracy_score(y_train, y_hat)
    # y_hat = clf.predict(X_test)
    # testAcc = accuracy_score(y_test, y_hat)
    # print("SVM training accuracy: {:.6f}".format(trainAcc))
    # print("SVM testing accuracy: {:.6f}".format(testAcc))


    # Create model
    model = createModel((X_train.shape[1],))
    model.summary()

    #Train model
    history = model.fit(X_train, y_train, epochs=1500, batch_size=128, validation_data=(X_test, y_test), shuffle=True, verbose=1)

    # Print Max accuracy
    print("Training final accuracy: {:0.6f}".format(history.history['acc'][-1]) )
    print("Testing final accuracy:  {:0.6f}".format(history.history['val_acc'][-1]) )
    print("Training max accuracy:   {:0.6f}".format(max(history.history['acc'])) )
    print("Testing max accuracy:    {:0.6f}".format(max(history.history['val_acc'])))

    # Plot accuracy
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(history.history['acc'], label='train acc')
    axes[0].plot(history.history['val_acc'], label='test acc')
    axes[1].plot(history.history['loss'], label='train loss')
    axes[1].plot(history.history['val_loss'], label='test loss')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    print('hello')