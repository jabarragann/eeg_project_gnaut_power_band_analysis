import SimpleLstmClassification as lstmClf
import numpy as np
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

def deep_ae(timesteps,features):

    networkInput = Input(shape=(timesteps, features))
    dense1 = Dense(75, activation="relu", use_bias=True, kernel_initializer="uniform")(networkInput)
    dense2 = Dense(25, activation="relu", use_bias=True, kernel_initializer="uniform")(dense1)
    dense3 = Dense(75, activation="relu", use_bias=True, kernel_initializer="uniform")(dense2)
    networkOutput = Dense(features, activation="sigmoid", use_bias=True, kernel_initializer="uniform")(dense3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam')

    return model1

#Global parameters
TIMESTEPS = 12
FEATURES = 150

if __name__ == '__main__':

    user = 'jhony'
    resultsPath = './results/'
    resultsFileName = resultsPath+'{:}_test.csv'.format(user)
    resultsContainer = []
    plotImageRootName = resultsPath + '{:}_test_'.format(user)

    path = './data/users/{:}/'.format(user)
    dataContainer = lstmClf.getDataSplitBySession(path)

    #Iterate over all the available sessions creating different testing sets.
    for testingKey in dataContainer.keys():
        print("testingKey")
        testX, testY = dataContainer[testingKey]['X'], dataContainer[testingKey]['y']
        trainX, trainY = [],[]

        #Use one session for testing and the rest of them as training.
        for trainingKey in dataContainer.keys():
            if trainingKey != testingKey:
                trainX.append(dataContainer[trainingKey]['X'])
                trainY.append(dataContainer[trainingKey]['y'])

        trainX = np.concatenate(trainX)
        trainY = np.concatenate(trainY)

        # Convert labels to one-hot encoding
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)

        #Create autoencoder
        autoencoder = deep_ae(trainX.shape[1],trainX.shape[2])
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        num_epochs = 2000
        batch_size = 256
        autoencoder_history = autoencoder.fit(trainX, trainX, validation_data=(testX, testX),
                                                epochs=num_epochs, batch_size=batch_size, shuffle=True)

        #Plot results
        fig, axes = plt.subplots(1, 1, sharex=True)
        axes.set_title('{:}_test_{:}'.format(user,testingKey))
        axes.plot(autoencoder_history.history['loss'], label='train acc')
        axes.plot(autoencoder_history.history['val_loss'], label='test acc')
        axes.legend()
        plt.show()

        break

    #
    #     #Save Plot
    #     plt.savefig(plotImageRootName+'{:}_.png'.format(testingKey))
    #
    #     #Add results
    #     data = {testingKey: [testingKey, max(history.history['val_acc']), history.history['val_acc'][-1]]}
    #     df1 = pd.DataFrame.from_dict(data, orient='index',columns=['test_session', 'max_test_acc', 'acc_at_final_epoch'])
    #     resultsContainer.append(df1)
    #
    # #Create final Dataframe
    # finalDf = pd.concat(resultsContainer)
    #
    # #Save results Dataframe to file
    # finalDf.to_csv(resultsFileName, index=False)



