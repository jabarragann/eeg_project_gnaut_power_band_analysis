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
    for user in ['jhony','juan','ryan','jackie']:
        resultsPath = './results/within-session2/'
        resultsFileName = resultsPath+'{:}_test.csv'.format(user)
        resultsContainer = []
        plotImageRootName = resultsPath + '{:}_test_'.format(user)

        path = './data/users/{:}/'.format(user)
        dataContainer = lstmClf.getDataSplitBySessionByTrial(path)

        try:
            #Repeat testing procedure
            for testingAttempt in range(4):
                pass
                #Select three sessions randomly
                arrTestSess = np.array(list(dataContainer.keys()))
                idx = np.arange(arrTestSess.shape[0])
                idx = np.random.choice(idx, 3, replace=False)
                arrTestSess = arrTestSess[idx]

                #Choose two trials randomly from each session
                arrTestTrial = np.array(list(product(lowTrials, highTrials)))
                idx = np.arange(arrTestTrial.shape[0])
                idx = np.random.choice(idx, 3, replace=True)
                arrTestTrial = arrTestTrial[idx]

                #Build test dict
                testDict = {}
                counter = 0
                for s in arrTestSess:
                    testDict[s] = arrTestTrial[counter]
                    counter += 1

                #Build datasets
                trainX, trainY = [], []
                testX, testY = [], []

                # Build testing set
                counter = 0
                for sessKey in dataContainer.keys():
                    if sessKey in testDict.keys():
                        for trialKey in dataContainer[sessKey].keys():
                            if trialKey ==  testDict[sessKey][0] or trialKey ==  testDict[sessKey][1]:
                                testX.append(dataContainer[sessKey][trialKey]['X'])
                                testY.append(dataContainer[sessKey][trialKey]['y'])
                                counter += 1

                #Build train set
                counter = 1
                for sessKey in dataContainer.keys():
                    if sessKey in testDict.keys():
                        for trialKey in dataContainer[sessKey].keys():
                            if trialKey !=  testDict[sessKey][0] and trialKey !=  testDict[sessKey][1]:
                                trainX.append(dataContainer[sessKey][trialKey]['X'])
                                trainY.append(dataContainer[sessKey][trialKey]['y'])
                            else:
                                counter += 1
                    else:
                        for trialKey in dataContainer[sessKey].keys():
                            trainX.append(dataContainer[sessKey][trialKey]['X'])
                            trainY.append(dataContainer[sessKey][trialKey]['y'])

                #Build Sets
                trainX = np.concatenate(trainX)
                trainY = np.concatenate(trainY)
                testX = np.concatenate(testX)
                testY = np.concatenate(testY)

                # Convert labels to one-hot encoding
                trainY = to_categorical(trainY)
                testY = to_categorical(testY)

                # Train Model
                history,model = trainTestModel(trainX, trainY, testX, testY, timesteps=trainX.shape[1], features=trainX.shape[2])

                # Plot results
                fig, axes = plt.subplots(2, 1, sharex=True)
                axes[0].set_title('{:}_test_{:}'.format(user, testingAttempt))
                axes[0].plot(history.history['acc'], label='train acc')
                axes[0].plot(history.history['val_acc'], label='test acc')
                axes[0].set_ylim([0.5, 1])
                axes[1].plot(history.history['loss'], label='train loss')
                axes[1].plot(history.history['val_loss'], label='test loss')
                axes[0].legend()
                axes[1].legend()

                # Save Plot
                plt.savefig(plotImageRootName + '{:}_.png'.format(testingAttempt))
                # plt.show()

                # Add results
                data = {testingAttempt: [testingAttempt, max(history.history['val_acc']),
                                         history.history['val_acc'][-1]]}
                df1 = pd.DataFrame.from_dict(data, orient='index',
                                             columns=['test_attempt', 'max_test_acc', 'acc_at_final_epoch'])
                resultsContainer.append(df1)
        except Exception as e:
            print("ERROR!!!!")
            print(e)
        finally:
            # Create final Dataframe
            finalDf = pd.concat(resultsContainer)

            # Save results Dataframe to file
            try:
                finalDf.to_csv(resultsFileName, index=False)
            except:
                finalDf.to_csv(resultsPath + 'temp_{:}.csv'.format(user), index=False)

            # print("Finally Clause")
