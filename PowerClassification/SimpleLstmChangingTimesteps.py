import SimpleLstmClassification as lstmClf
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

#Global parameters
# TIMESTEPS = 12
FEATURES = 120

def trainTestModel(X_train, y_train, X_test, y_test, timesteps=12):
    # Normalize data
    X_mean = np.mean(X_train, axis=(0, 1))
    X_train = (X_train - X_mean)
    X_test = (X_test - X_mean)

    # Print data shape
    print("Train Shape", X_train.shape)
    print("Test Shape", X_test.shape)

    # Train model
    batch = 256
    epochs = 400
    model = lstmClf.lstm2(*(timesteps, FEATURES))

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_test, y_test))

    return history

if __name__ == '__main__':

    user = 'juan'
    resultsPath = './results/'
    resultsFileName = resultsPath+'{:}_test.csv'.format(user)
    resultsContainer = []
    plotImageRootName = resultsPath + '{:}_test_'.format(user)

    path = './data/users/{:}/'.format(user)

    # Loop over the different Timesteps
    for lstmSteps in [4, 8, 12, 16, 20, 24]:
        dataContainer = lstmClf.getDataSplitBySession(path, timesteps=lstmSteps)

        #Iterate over all the available sessions creating different testing sets.
        for testingKey in dataContainer.keys(): #['4','5','6','7']:

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


            #Train Model
            history = trainTestModel(trainX,trainY,testX,testY,timesteps=lstmSteps)

            #Plot results
            fig, axes = plt.subplots(2, 1, sharex=True)
            axes[0].set_title('{:}_test_{:}_timesteps_{:}'.format(user,testingKey,str(lstmSteps)))
            axes[0].plot(history.history['acc'], label='train acc')
            axes[0].plot(history.history['val_acc'], label='test acc')
            axes[0].set_ylim([0.5, 1])
            axes[1].plot(history.history['loss'], label='train loss')
            axes[1].plot(history.history['val_loss'], label='test loss')
            axes[0].legend()
            axes[1].legend()
            # plt.show()
            #Save Plot and close plot
            plt.savefig(plotImageRootName+'{:}_timesteps_{:02d}_.png'.format(testingKey,lstmSteps))
            plt.close(fig)

            #Add results
            data = {testingKey: [testingKey,lstmSteps, max(history.history['val_acc']), history.history['val_acc'][-1]]}
            df1 = pd.DataFrame.from_dict(data, orient='index',columns=['test_session', 'timesteps', 'max_test_acc', 'acc_at_final_epoch'])
            resultsContainer.append(df1)

    #Create final Dataframe
    finalDf = pd.concat(resultsContainer)

    #Save results Dataframe to file
    finalDf.to_csv(resultsFileName, index=False)