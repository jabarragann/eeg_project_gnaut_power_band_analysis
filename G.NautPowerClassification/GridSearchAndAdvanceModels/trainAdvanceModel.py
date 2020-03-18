from GridSearch import getData

import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization,Softmax,LSTM, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers, constraints

#Lstm mode to test
def createLstmModel(isBidirectional=True, inputLayerNeurons= 8, inputLayerDropout=0.5, lstmSize = 4):
    timesteps = 12
    features = 150

    #Input layer
    networkInput = Input(shape=(timesteps, features))
    dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

    #First Hidden layer
    hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    #Choose if the network should be bidirectional
    if isBidirectional:
        lstmLayer = LSTM(lstmSize, stateful=False,
                         dropout=0.5, kernel_regularizer=regularizers.l2(0.05))
        #hidden2 = Bidirectional( LSTM(lstmSize, stateful=False, dropout=0.5), merge_mode='concat' ) (batchNorm1)
        hidden2 = Bidirectional( lstmLayer, merge_mode='concat' ) (batchNorm1)
    else:
        hidden2 = LSTM(lstmSize, stateful=False, dropout=0.5) (batchNorm1)

    hidden3 = Dense(2, activation='linear')(hidden2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1

if __name__ == '__main__':

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # Load data
    X,Y = getData()
    Y = to_categorical(Y)

    trainX,testX,trainY,testY = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=seed)

    #Normalize data
    globalMean = np.mean(trainX,axis= (0,1))
    globalStd = np.std(trainX,  axis= (0,1))

    trainX =  (trainX-globalMean)/(globalStd + 1e-18)
    testX =   (testX-globalMean)/(globalStd + 1e-18)

    # create model
    model = createLstmModel(isBidirectional=True, inputLayerNeurons= 64, inputLayerDropout=0.3, lstmSize = 4)

    #Train model
    history = model.fit(trainX,trainY,validation_data=(testX,testY), epochs=120, batch_size=512, verbose=1)

    print("Maximum accuracy: {:0.5f} in epoch {:d}".format(max(history.history['val_acc']), np.argmax(history.history['val_acc'])))
    #plot
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(history.history['acc'], label='train acc')
    axes[0].plot(history.history['val_acc'], label='test acc')
    axes[0].set_ylim([0.5, 1])
    axes[1].plot(history.history['loss'], label='train loss')
    axes[1].plot(history.history['val_loss'], label='test loss')
    axes[0].legend()
    axes[1].legend()

    plt.show()