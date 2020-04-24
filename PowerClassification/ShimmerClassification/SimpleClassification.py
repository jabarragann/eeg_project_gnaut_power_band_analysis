from pathlib import Path
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import seaborn as sns

def getData(userFilter = None):
    dataContainer = []
    for file in DATA_DIR.rglob('*.txt'):
        df = pd.read_csv(file, sep=',', )
        dataContainer.append(copy.deepcopy(df))

    dataContainer = pd.concat(dataContainer, ignore_index=True)

    if not userFilter is None:
        dataContainer = dataContainer.loc[dataContainer['User'] == userFilter]

    return dataContainer

def createModel(inputShape):
    regularizer = None
    inputFeatures = Input(shape=inputShape)
    dropout1 = Dropout(rate=0.02)(inputFeatures)
    hidden1 = Dense(25, activation='relu', kernel_regularizer=regularizer)(dropout1)
    batchNorm1 = BatchNormalization()(hidden1)

    dropout2 = Dropout(rate=0.1)(batchNorm1)
    hidden2 = Dense(15, activation='relu', kernel_regularizer=regularizer)(dropout2)
    batchNorm2 = BatchNormalization()(hidden2)

    dropout4 = Dropout(rate=0.1)(batchNorm2)
    hidden4 = Dense(2)(dropout4)

    output = Softmax()(hidden4)
    model = Model(inputs=inputFeatures, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

if __name__== '__main__':

    #metrics = ["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50", "hr_mad", "sd1", "sd2", "s", "sd1/sd2"]
    metrics = ["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50", "hr_mad", "sd1", "sd2", "s"]
    DATA_DIR = Path('./').resolve().parent / 'data' / 'shimmerPreprocessed' / '60s'
    print('Data Directory')
    print(DATA_DIR)

    data = getData()
    xData = data[metrics].values
    yData = data['label'].values

    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.33, random_state=42, shuffle=True)

    #Clip outliers
    per25 = np.percentile(xTrain, 25, axis=0, keepdims=1)
    per75 = np.percentile(xTrain, 75, axis=0, keepdims=1)
    iqr1 = per75-per25
    maxValues = per75+1.5*iqr1
    minValues = per25-1.5*iqr1
    xTrain = np.clip(xTrain,minValues,maxValues)
    xTest = np.clip(xTest, minValues, maxValues)
    #Normalize values
    globalMean = xTrain.mean(axis=0)
    globalStd = xTrain.std(axis=0)
    xTrain = (xTrain - globalMean)/(globalStd+ 10e-10)
    xTest =  (xTest  - globalMean)/(globalStd+ 10e-10)
    #One-hot encoding
    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)

    model = createModel(xTrain.shape[1])
    model.summary()

    history = model.fit(xTrain, yTrain,batch_size=32,epochs=300,validation_data=(xTest, yTest), verbose=0)
    testEvaluation =  model.evaluate(xTest,yTest)[1]
    print("Final test accuracy {:0.5f}".format(testEvaluation))
    print("Max test accuracy {:0.5f}".format(max(history.history['val_acc'])))

    #Confusion Matrix
    confusionMatrix = np.zeros((2,2))
    predicted = model.predict(xTest)
    predicted = np.argmax(predicted,axis=1)
    yTest = np.argmax(yTest, axis=1)
    for i, j in zip(predicted, yTest):
        confusionMatrix[i.item(), j.item()] += 1

    confusionMatrix = confusionMatrix
    testEvaluation2 =  np.sum(yTest == predicted)
    confusionMatrix = pd.DataFrame(data=confusionMatrix,
                                   index=["Predicted low", "Predicted high"],
                                   columns=["Actual low", "Actual high"])

    print("Test Accuracy {:0.5f}".format(testEvaluation2))
    print("Confusion matrix")
    print(confusionMatrix)

    fig,axes = plt.subplots(3)
    axes[0].plot(history.history['acc'], label='train')
    axes[0].plot(history.history['val_acc'], label='test')
    axes[0].legend()
    axes[1].plot(history.history['loss'],label='train')
    axes[1].plot(history.history['val_loss'],label='test')
    axes[1].legend()
    #"0.2%"
    sns.heatmap(confusionMatrix, annot=True, fmt="0.2f", linewidths=.5, ax=axes[2])
    plt.tight_layout()
    plt.show()





