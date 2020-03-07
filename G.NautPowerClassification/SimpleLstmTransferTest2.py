import SimpleLstmClassification as lstmClf
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pandas as pd
from itertools import product

#Global variables

lowTrials  = ['1','3','5']
highTrials = ['2','4','6']


def trainTestModel(X_train, y_train, X_test, y_test, features=None, timesteps=None, lr=None):
    # Normalize data
    X_mean = np.mean(X_train, axis=(0, 1))
    X_train = (X_train - X_mean)
    X_test = (X_test - X_mean)

    # Print data shape
    print("Train Shape", X_train.shape)
    print("Test Shape", X_test.shape)

    # Train model
    batch = 256
    epochs = 250
    model = lstmClf.lstm2(*(timesteps, features))

    if lr is not None:
        print("Change Learning rate to ", lr)
        optim = Adam(learning_rate = lr)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])


    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test))

    return history, model

if __name__ == '__main__':

    # user = 'juan'
    for user in ['jhony','jackie','ryan','juan']:
        resultsPath = './results_transfer2/'
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
                idx = np.random.choice(idx, 3, replace=False)

                for lt,ht in arr[idx]:
                    # Get testing trials and transfer trials
                    testTrials = np.array(list(dataContainer[testingKey].keys()))
                    idx =  np.argwhere(testTrials == lt), np.argwhere(testTrials == ht)
                    testTrials = np.delete(testTrials, idx)
                    transferTrials = [lt, ht]

                    transferX = np.concatenate([dataContainer[testingKey][i]['X'] for i in transferTrials])
                    transferY = np.concatenate([dataContainer[testingKey][i]['y'] for i in transferTrials])
                    testX = np.concatenate([dataContainer[testingKey][i]['X'] for i in testTrials])
                    testY = np.concatenate([dataContainer[testingKey][i]['y'] for i in testTrials])

                    # Use the rest of the sessions for training.
                    trainX, trainY = [], []
                    for trainingKey in dataContainer.keys():
                        for trialKey in dataContainer[trainingKey].keys():
                            if trainingKey != testingKey:
                                trainX.append(dataContainer[trainingKey][trialKey]['X'])
                                trainY.append(dataContainer[trainingKey][trialKey]['y'])

                    trainX = np.concatenate(trainX)
                    trainY = np.concatenate(trainY)

                    #Augment transfer set by injecting random noise
                    augmentedTraining = transferX + 0.4 * np.random.randn(*transferX.shape)
                    transferX = np.concatenate((transferX, augmentedTraining))
                    transferY = np.concatenate((transferY, transferY))

                    # Convert labels to one-hot encoding
                    trainY = to_categorical(trainY)
                    transferY = to_categorical(transferY)
                    testY = to_categorical(testY)

                    # Train Model
                    history, model = trainTestModel(trainX, trainY, testX, testY, timesteps=trainX.shape[1], features=trainX.shape[2])
                    K.clear_session()

                    # Plot results
                    fig, axes = plt.subplots(2, 1, sharex=True)
                    axes[0].set_title('{:}_test_{:}_{:}_{:}_before'.format(user, testingKey,lt,ht))
                    axes[0].plot(history.history['acc'], label='train acc')
                    axes[0].plot(history.history['val_acc'], label='test acc')
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
                    transferHistory, model = trainTestModel(trainX, trainY, testX, testY, timesteps=trainX.shape[1], features=trainX.shape[2])
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
                    data = {testingKey: [testingKey,lt,ht, history.history['val_acc'][-1], transferHistory.history['val_acc'][-1], difference]}
                    df1 = pd.DataFrame.from_dict(data, orient='index',
                                                 columns=['test_session', 'low_trial','high_trial','test_acc_before', 'test_acc_after','Difference'])
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
