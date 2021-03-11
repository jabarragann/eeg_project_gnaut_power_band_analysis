import sys
sys.path.append('./../../')

from pathlib import Path
import re
import numpy as np
from itertools import product
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

#Tensorflow utils
from SpectralImagesClassification.LoadDataUtils import createImageDataset
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, Input, BatchNormalization, Softmax
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

def extract_file_info(file_name):
    uid = re.findall('.+(?=_S[0-9]{2}_T[0-9]{2}_)', file_name)[0]
    session = int(re.findall('(?<=_S)[0-9]+(?=_T[0-9]{2}_)', file_name)[0])
    trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]{2}(?=_)', file_name)[0])
    task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_raw\.edf)', file_name)[0]
    return session, trial

def concat_data(blocks, data_dict, cv_block):
    X_list = []
    y_list = []
    for b in blocks:
        for trial in cv_block[b]:
            X_list.append(data_dict[trial]['X'])
            y_list.append(data_dict[trial]['y'])
    return np.concatenate(X_list), np.concatenate(y_list)

def prepare_data_for_training(x_train,y_train,x_test,y_test, num_classes=2):
    ##Global frame/time normalization
    global_mean = x_train.mean();
    global_std = x_train.std();
    x_train_2 = (x_train - global_mean) / global_std;
    x_test_2 = (x_test - global_mean) / global_std;

    ## convert class vectors to binary class matrices
    y_train_2 = keras.utils.to_categorical(y_train, num_classes)
    y_test_2 = keras.utils.to_categorical(y_test, num_classes)

    x_train_2 = x_train_2.astype('float32')
    x_test_2 = x_test_2.astype('float32')

    print("Train", x_train.shape, y_train.shape)
    print("Test", x_test.shape, y_test.shape)
    return x_train_2,y_train_2,x_test_2,y_test_2

def createLstmModel(input_shape, num_classes, lstmLayers=1, lstmOutputSize=4.0,
                    isBidirectional=0.0, inputLayerNeurons=10, inputLayerDropout=0.55, threshold=0.5):
    dropoutRate = inputLayerDropout

    lstmLayers = int(lstmLayers)
    lstmOutputSize = int(lstmOutputSize)
    isBidirectional = int(isBidirectional)

    # Input layer
    networkInput = Input(shape=input_shape)
    dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

    # First Hidden layer
    hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
    dropout2 = Dropout(rate=dropoutRate)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    out = batchNorm1
    for i in range(1, lstmLayers + 1):
        retSeq = False if i == lstmLayers else True
        lstmLayer = LSTM(lstmOutputSize, stateful=False, return_sequences=retSeq,
                         dropout=dropoutRate, kernel_regularizer=regularizers.l2(1.0))
        if isBidirectional:
            out = Bidirectional(lstmLayer, merge_mode='concat')(out)
        else:
            out = lstmLayer(out)

    hidden3 = Dense(num_classes, activation='linear')(out)
    networkOutput = Softmax()(hidden3)

    m = keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=threshold)
    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam',
                   metrics=['accuracy'])#[m]

    return model1

def main():
    # #Args
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_name", required=True, help="Name of folder where the data is going to be saved.")
    # parser.add_argument("--segment_length", required=True, type=float)
    # args = parser.parse_args()

    dstPath = Path("automated_results") /  "last_try_juan" ##args.experiment_name
    trainPlotsPath = dstPath / "train_plots"

    if dstPath.exists():
        print('experiment exists!')
        exit(0)
    else:
        trainPlotsPath.mkdir(parents=True)


    batch_size = 64
    epochs = 200
    th = 0.5

    #Parameters
    image_size = 32
    frame_length = 1 #args.segment_length
    sequence_length = 25  # 20
    overlap = 0.5
    num_classes = 2
    cv_blocks = {'b1': [1, 2], 'b2': [3,4 ],
                 'b3': [5, 6],'b4': [7, 8], 'b5': [9, 10] }

    # test_block = {'b5': [9, 10],'b4': [7, 8], 'b5': [13, 15] }
    cv_data = {}

    # Load all data
    dataPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\edf\Juan\S6")
    for f in dataPath.rglob("*.edf"):
        print(f.name)
        session, trial = extract_file_info(f.name)
        X,y  = createImageDataset([f],frame_duration=frame_length,overlap=overlap,lstm_format=True,
                                  lstm_sequence_length=sequence_length,sequence_overlap=True,read_from='files',
                                  file_name_format=2,labels = ["NeedlePassing", "BloodNeedle"])
        cv_data[trial] = {'X':X, 'y':y}

    # Cross validation Routine
    lstm_seq = [sequence_length]
    dropout_rates = [0.5]
    total_combinations = len(lstm_seq) * len(dropout_rates)
    combination = product(lstm_seq, dropout_rates)

    # Results table
    blocks = list(cv_blocks.keys())
    idx = pd.MultiIndex.from_product([range(total_combinations), blocks])
    df = pd.DataFrame(index=idx, columns=['trainAcc', 'valAcc', 'TP', 'TN', 'FP', 'FN','samples', 'lstmLength', 'dropout'])
    file_to_save = "./cross-validation-results_s{:}.csv".format(frame_length)

    for i, params in enumerate(combination):
        s_len, dropout = params
        input_shape = (s_len, 90)

        # Cross validation routine
        for k, v in cv_blocks.items():
            model = createLstmModel(input_shape, num_classes, inputLayerDropout=dropout, threshold=th)

            #Create data
            trainingBlocks = list(cv_blocks.keys())
            trainingBlocks.remove(k)
            validationBlocks = [k]
            print(trainingBlocks)
            print(validationBlocks)

            x_train, y_train = concat_data(trainingBlocks,cv_data,cv_blocks)
            x_test, y_test = concat_data(validationBlocks,cv_data,cv_blocks)

            x_train, y_train, x_test, y_test = prepare_data_for_training(x_train,y_train,x_test,y_test)

            #Train model
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(x_test, y_test),
                                shuffle=True)

            plotTitle = "cv_b_{:}_seg_{:}_lstmseg_{:}".format(k,frame_length,sequence_length)
            plt.plot(history.history['accuracy'],label="train")
            plt.plot(history.history['val_accuracy'], label='val')
            plt.legend()
            plt.savefig(trainPlotsPath / (plotTitle+'.png') )
            plt.close()

            #Test model
            trainAcc = history.history['accuracy'][-1] #"binary_accuracy"
            testAcc = history.history['val_accuracy'][-1]
            test_pred = model.predict(x_test)
            predictions = np.argmax(test_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            conf_mat = confusion_matrix(y_test, predictions)

            # save in df
            df.loc[(i, k), ('lstmLength', 'dropout')] = params
            df.loc[(i, k), ('trainAcc', 'valAcc')] = [trainAcc,testAcc]
            df.loc[(i, k), ('TN', 'FP', 'FN', 'TP')] = conf_mat.ravel() #Taken from sklearn documentation
            df.loc[(i, k), 'samples'] = y_test.shape[0]

            df.to_csv(dstPath / file_to_save)

    df.to_csv(dstPath / file_to_save)

if __name__ == "__main__":
    main()
