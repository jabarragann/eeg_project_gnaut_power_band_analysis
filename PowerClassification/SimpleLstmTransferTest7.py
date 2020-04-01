'''
In the following script, I am going to test if the having multiple users in the training set will boost the classification results
in the cross-session setting.

Constraints of the experiment:
1)Only data from one user will be in the testing set.
2)The testing set will be a single session from the testing user.
3)Another session of the testing user is added to the validation set.
4)The rest of the sessions of the testing user go to the training set.

Training and validation have data from multiple users.

The experiment is going to be repeated for every session of the testing user.

Conclusion:
'''

import SharedModules.mod2 as m
from SharedModules import mod2 as mod2Sh
from PowerClassification.test import test

# import test.test as t
# import test.mod2 as mod2
# import SimpleLstmClassification as lstmClf
# import numpy as np
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import backend as K
# import pandas as pd
# import copy
# import tensorflow.keras as keras
# from itertools import product
# import os

test.testFunc()
mod2Sh.mod2()
# t.testFunc()
# mod2.mod2()
# mod2Sh.mod2()

# #Global variables
#
# lowTrials  = ['1','3','5']
# highTrials = ['2','4','6']
#
# # Train again with the transfer data
# EPOCHS = 300
# MINIBATCH = 512
# TRANSFER_EPOCHS = 250
# TRANSFER_MINIBATCH = 256
# MODEL_NUMBER = 2
#
# class NetworkTrainingModule:
#
#     def __init__(self):
#         pass
#
#     def makeDir(path):
#         try:
#             os.mkdir(path)
#         except Exception as e:
#             print(e)
#
#     def trainModelEarlyStop(self, model, X_train, y_train,
#                             X_val, y_val, X_test, y_test,
#                             batchSize=256*2, epochs=300, debug=True):
#         '''
#             The following function trains a model with the given train data and validation data.
#             At the same time it checks the performance of the model in the testing set through the
#             EarlyStoppingCallback class. After training, the model is returned to the step in time
#             where the best validation accuracy is obtained.
#
#             :param model:
#             :param X_train:
#             :param y_train:
#             :param X_val:
#             :param y_val:
#             :param X_test:
#             :param y_test:
#             :param batchSize: 512
#             :param epochs: 300
#             :param debug: True
#
#             :return: kerasHistory, model, earlyStoppingInstance
#         '''
#
#         # Normalize data
#         # X_mean = np.mean(X_train, axis=(0,1))
#         # X_std = np.std(X_train, axis = (0,1))
#
#         # Print data shape
#         if debug:
#             print("Train Shape", X_train.shape)
#             print("Test Shape", X_test.shape)
#
#         # Training parameters
#         batchSize = batchSize
#         epochs = epochs
#         earlyStopCallback = self.EarlyStoppingCallback(2, additionalValSet=(X_test, y_test))
#
#         #Train model
#         callbacks = [earlyStopCallback]
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize,
#                             validation_data=(X_val, y_val), callbacks=callbacks)
#
#         #Get best weights
#         model.set_weights(earlyStopCallback.bestWeights)
#
#         return history, model, earlyStopCallback
#
#     @staticmethod
#     def createPlot(trainingHistory, title, path, show_plot=False, earlyStopCallBack=None):
#         '''
#         Create plots from training history
#         :param trainingHistory:
#         :param title:
#         :param path:
#         :param show_plot: Default=False
#         :param earlyStopCallBack: Default = None
#         :return:
#         '''
#
#         fig, axes = plt.subplots(2, 1, sharex=True)
#         axes[0].set_title(title)
#         axes[0].plot(trainingHistory.history['acc'], label='train acc')
#         axes[0].plot(trainingHistory.history['val_acc'], label='val acc')
#         axes[0].set_ylim([0.5, 1])
#         axes[1].plot(trainingHistory.history['loss'], label='train loss')
#         axes[1].plot(trainingHistory.history['val_loss'], label='val loss')
#         axes[0].legend()
#         axes[1].legend()
#
#         #Testing acc
#         if earlyStopCallBack:
#             axes[0].plot(earlyStopCallBack.validationAcc2, label='test acc')
#             axes[0].axvline(x=earlyStopCallBack.epochOfMaxValidation)
#
#         if show_plot:
#             plt.show()
#
#         plt.savefig(path)
#         plt.close(fig)
#
# allUsers = ['juan','ryan','jhony','jackie',]
#
# if __name__ == '__main__':
#
#     #Create Network training module
#     trainingModule = NetworkTrainingModule()
#
#     #Final container with the data of all users.
#     resultsContainer2 = []
#
#     resultsPath = './results/results_transfer7/'
#
#     #Iterate over all the user, so that the testing user will change in every iteration
#     for testUser in ['juan','jackie','ryan','jhony']:
#
#         resultsFileName = resultsPath+'{:}_test.csv'.format(testUser)
#         resultsContainer = []
#         plotImageRootName = resultsPath + '{:}_test_'.format(testUser)
#
#         # Set of data - either normal (users) or eeglab (usersEEGLab)
#         pBase = 'usersEEGLab'
#
#         #Get test user data
#         path = './data/{:}/{:}/'.format(pBase,testUser)
#         testUserContainer = lstmClf.getDataSplitBySessionByTrial(path)
#
#         #Build training and validation sets
#
#         #Initially remove testing user from training and create train set without him
#         trainingUsers = copy.copy(allUsers)
#         trainingUsers.remove(testUser)
#
#         firstTrainX = []
#         firstTrainY = []
#         firstValX = []
#         firstValY = []
#         logger = [[],[],[]]
#         for u in trainingUsers:
#             #Open user data
#             p = './data/{:}/{:}/'.format(pBase, u)
#             dataContainer = lstmClf.getDataSplitBySession(p)
#
#             #Choose randomly 4 sessions for testing and 1 for validation
#             availableSessions = np.array(list(dataContainer.keys()))
#             idx = np.arange(len(availableSessions))
#             idx = np.random.choice(idx, 4, replace=False)
#             trSess = availableSessions[idx[:-1]]
#             vlSess = availableSessions[idx[ -1]]
#
#             #create sets
#             firstTrainX.append(np.concatenate([dataContainer[i]['X'] for i in trSess]))
#             firstTrainY.append(np.concatenate([dataContainer[i]['y'] for i in trSess]))
#             firstValX.append(np.concatenate([dataContainer[i]['X'] for i in vlSess]))
#             firstValY.append(np.concatenate([dataContainer[i]['y'] for i in vlSess]))
#
#             logger[0].append(u), logger[1].append(trSess), logger[2].append(vlSess)
#
#         try:
#             #Iterate over all the sessions of the testing user
#             #Use one session for the test set and one for validation and the rest for training
#
#             #Transform default dict to dict to avoid modifications of the container
#             testUserContainer = dict(testUserContainer)
#
#             for testingKey in testUserContainer.keys():
#
#                 #copy data from the other users
#                 trainX = copy.copy(firstTrainX)
#                 trainY = copy.copy(firstTrainY)
#                 valX = copy.copy(firstValX)
#                 valY = copy.copy(firstValY)
#
#                 allSessions =  list(testUserContainer.keys())
#                 allSessions.remove(testingKey)
#                 allSessions = np.array(allSessions)
#
#                 #Randomly organize the available sessions, then grab the first one for validation and the rest for training
#                 idx = np.arange(allSessions.shape[0])
#                 idx = np.random.choice(idx, len(idx), replace=False)
#                 allSessions = allSessions[idx]
#
#                 #Add randomly sampled session to the validation set
#                 [valX.append(testUserContainer[allSessions[0]][i]['X']) for i in testUserContainer[allSessions[0]].keys()]
#                 [valY.append(testUserContainer[allSessions[0]][i]['y']) for i in testUserContainer[allSessions[0]].keys()]
#
#                 #Add the remaining sessions to the training set
#                 for j in allSessions[1:]:
#                     [trainX.append(testUserContainer[j][i]['X']) for i in testUserContainer[j].keys()]
#                     [trainY.append(testUserContainer[j][i]['y']) for i in testUserContainer[j].keys()]
#
#                 #Concatenate all sessions for validation and training
#                 trainX = np.concatenate(trainX)
#                 trainY = np.concatenate(trainY)
#                 valX = np.concatenate(valX)
#                 valY = np.concatenate(valY)
#
#                 # Sample randomly some trials to build the transfer subset.
#                 arr = np.array(list(product(lowTrials, highTrials)))
#                 idx = np.arange(arr.shape[0])
#                 idx = np.random.choice(idx, 1, replace=False)
#
#                 for lt, ht in arr[idx]:
#                     # Get testing trials and transfer trials
#                     testTrials = np.array(list(testUserContainer[testingKey].keys()))
#                     idx = np.argwhere(testTrials == lt), np.argwhere(testTrials == ht)
#                     testTrials = np.delete(testTrials, idx)
#                     transferTrials = [lt, ht]
#
#                     transferX = np.concatenate([testUserContainer[testingKey][i]['X'] for i in transferTrials])
#                     transferY = np.concatenate([testUserContainer[testingKey][i]['y'] for i in transferTrials])
#                     testX = np.concatenate([testUserContainer[testingKey][i]['X'] for i in testTrials])
#                     testY = np.concatenate([testUserContainer[testingKey][i]['y'] for i in testTrials])
#
#                     # Augment transfer set by injecting random noise
#                     augmentedTraining = transferX + 0.15 * np.random.randn(*transferX.shape)
#                     transferX = np.concatenate((transferX, augmentedTraining))
#                     transferY = np.concatenate((transferY, transferY))
#
#                     # Convert labels to one-hot encoding
#                     trainY = to_categorical(trainY)
#                     transferY = to_categorical(transferY)
#                     testY = to_categorical(testY)
#                     valY = to_categorical(valY)
#
#                     # Normalize data - with training set parameters
#                     globalMean = trainX.mean(axis=(0, 1))
#                     globalStd = trainX.std(axis=(0, 1))
#
#                     trainX = (trainX - globalMean) / (globalStd + 1e-18)
#                     valX = (valX - globalMean) / (globalStd + 1e-18)
#                     transferX = (transferX - globalMean) / (globalStd + 1e-18)
#                     testX = (testX - globalMean) / (globalStd + 1e-18)
#
#
#                     #Create model
#                     # model = lstmClf.lstm2(*(trainX.shape[1], trainX.shape[2]))
#                     model = lstmClf.createAdvanceLstmModel(*(trainX.shape[1], trainX.shape[2]))
#
#                     # Train Model - First round
#                     history, model, earlyStopping = trainingModule.trainModelEarlyStop(
#                                                     model, trainX, trainY, valX, valY, testX, testY)
#
#                     #Evaluate Model
#                     evalTestBefore = model.evaluate(testX, testY)
#                     K.clear_session()
#
#                     #Create plots
#                     title = '{:}_test_{:}_{:}_{:}_before'.format(testUser, testingKey, lt, ht)
#                     graphPath = plotImageRootName + '{:}_{:}_{:}_a_before.png'.format(testingKey, lt, ht)
#                     trainingModule.createPlot(history,title, graphPath, earlyStopCallBack=earlyStopping)
#
#                     # # Plot results
#                     # fig, axes = plt.subplots(2, 1, sharex=True)
#                     # axes[0].set_title()
#                     # axes[0].plot(history.history['acc'], label='train acc')
#                     # axes[0].plot(history.history['val_acc'], label='val acc')
#                     # axes[0].plot(earlyStopping.validationAcc2, label = 'test acc')
#                     # axes[0].axvline(earlyStopping.epochOfMaxValidation, 0, 1)
#                     # axes[0].set_ylim([0.5, 1])
#                     # axes[1].plot(history.history['loss'], label='train loss')
#                     # axes[1].plot(history.history['val_loss'], label='test loss')
#                     # axes[0].legend()
#                     # axes[1].legend()
#                     # # plt.show()
#                     #
#                     # # Save Plot
#                     # plt.savefig(plotImageRootName + '{:}_{:}_{:}_a_before.png'.format(testingKey, lt, ht))
#                     # plt.close(fig)
#
#                     evalTestAfter = model.evaluate(testX, testY)
#                     K.clear_session()
#
#                     #Save all the results
#                     data = {testUser: [pBase, testUser, logger[0], " ".join([str(t) for t in logger[1]])
#                                         , logger[2], testingKey,lt,ht
#                                         ,max(history.history['val_acc']), evalTestBefore[1], evalTestAfter[1]]}
#
#                     df1 = pd.DataFrame.from_dict(data, orient='index',
#                                                  columns=['preprocess','user','others','others_train','others_val','test_session',
#                                                           'low_trial', 'high_trial','validation', 'testBefore','testAfter'])
#                     resultsContainer.append(df1)
#                     resultsContainer2.append(df1)
#                     valResults = 0
#
#                     K.clear_session()
#
#         finally:
#             # Create final Dataframe
#             finalDf = pd.concat(resultsContainer)
#
#             # Save results Dataframe to file
#             try:
#                 finalDf.to_csv(resultsFileName, index=False)
#             except:
#                 finalDf.to_csv(resultsPath + 'temp_{:}.csv'.format(testUser), index=False)
#
#             # print("Finally Clause")
#
#     finalDf = pd.concat(resultsContainer2)
#     finalDf.to_csv(resultsPath + 'complete.csv', index=False)