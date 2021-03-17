import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
import EyeTrackerClassification.EyeTrackerUtils as etu
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\eyetracker\S2")
    labels = ['NeedlePassing', 'BloodNeedle']

    files_dict = etu.search_files_on_path(path)
    files_dict = etu.merge_files(files_dict, labels)

    #split into train val
    # valList = [1,2]#[3,4]#[5,6]#[7,8]#[9,10]
    for valList in [[1,2],[3,4],[5,6],[7,8],[9,10]]:
        break
        train_files = [files_dict[key] for key in files_dict.keys() if key not in valList]
        val_files = [files_dict[key] for key in files_dict.keys() if key in valList]

        train_x, train_y = etu.get_data(train_files)
        val_x, val_y = etu.get_data(val_files)

        train_x, val_x = train_x.values,val_x.values
        train_y, val_y = train_y.values, val_y.values

        #Normalize
        global_mean = train_x.mean(axis=0).reshape(1,5)
        global_std = train_x.std(axis=0).reshape(1,5)
        train_x = (train_x - global_mean) / global_std
        val_x = (val_x - global_mean) / global_std

        #Create model
        model = create_baseline()
        #Train
        history = model.fit(train_x, train_y,
                        batch_size=20,
                        epochs=120,
                        validation_data=(val_x,val_y),
                        shuffle=True,
                        verbose=False)

        # from sklearn import svm
        # clf = svm.SVC(kernel='rbf')
        # clf.fit(train_x, train_y)
        # svm_pred = clf.predict(val_x)
        # svm_acc = sum(val_y.squeeze() == svm_pred.squeeze()) / val_y.shape[0]

        #predict
        predicted = model.predict(val_x)
        predicted_label = predicted.copy()
        predicted_label[predicted > 0.5] = 1.0
        predicted_label[predicted < 0.5] = 0.0
        acc1 = sum(val_y.squeeze() == predicted_label.squeeze())/val_y.shape[0]
        acc2 = model.evaluate(val_x,val_y,verbose=0)[1]

        print("Shapes: ", train_x.shape, val_x.shape)
        print(valList, "nn_acc")
        print("{:0.3f}".format(acc2))

        # plotTitle = "eye tracker"
        # plt.plot(history.history['accuracy'], label="train")
        # plt.plot(history.history['val_accuracy'], label='val')
        # plt.legend()
        # plt.show()
        # plt.close()

    ##############################
    # Train a model with all data#
    ##############################
    train_files = [files_dict[key] for key in files_dict.keys()]
    train_x, train_y = etu.get_data(train_files)
    train_x = train_x.values
    train_y = train_y.values
    # Normalize
    global_mean = train_x.mean(axis=0).reshape(1, 5)
    global_std = train_x.std(axis=0).reshape(1, 5)
    train_x = (train_x - global_mean) / global_std
    # Create model
    print("train shape", train_x.shape)
    model = create_baseline()
    # Train
    history = model.fit(train_x, train_y,
                        batch_size=20,
                        epochs=120,
                        shuffle=True,
                        verbose=True)
    model.save('./model/model.h5')
    normalizer = {'mean':global_mean,'std':global_std}
    pickle.dump(normalizer, open('./model/normalizer.pic','wb'))

if __name__ == "__main__":
    main()