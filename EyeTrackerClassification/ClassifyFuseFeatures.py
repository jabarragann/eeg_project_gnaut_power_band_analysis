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


def create_baseline(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_files(path,labels):
    files_dict = {}
    for file in path.rglob("*.txt"):
        task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_fuse)', file.name)[0]
        trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]{2}(?=_)', file.name)[0])
        label = 1.0 if labels[1]==task else 0.0
        print(file.name,task,label)
        X = pd.read_csv(file,index_col=[0])
        y = np.ones(X.shape[0]) * label
        files_dict[trial] = {'X':X.drop('LSL_TIME',axis=1).values,'y':y}

        assert X.shape[0] == y.shape[0], "Dimension error in file {:}".format(file.name)
    return files_dict

def merge_files(files_dict):
    X, y = [],[]
    for fx,fy in files_dict:
        X.append(fx)
        y.append(fy)
    return np.concatenate(X), np.concatenate(y)

def main():
    model_name = 'juan-9-feat'
    single_user = False
    number_of_feat = 9
    path_main_u = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJuan\S7")

    path2 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S2")
    path3 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UKeyu\S1")

    labels = ['NeedlePassing', 'BloodNeedle']

    files_main_u = load_files(path_main_u,labels)
    files_dict2 = load_files(path2,labels)
    files2 = [[files_dict2[key]['X'], files_dict2[key]['y']] for key in files_dict2.keys()]
    files_dict3 = load_files(path3, labels)
    files3 = [[files_dict3[key]['X'], files_dict3[key]['y']] for key in files_dict3.keys()]
    files_other_u = files2+files3

    #split into train val
    for valList in [[1,2],[3,4],[5,6],[7,8],[9,10]]:
        # break
        train_files = [ [ files_main_u[key]['X'],files_main_u[key]['y']] for key in files_main_u.keys() if key not in valList]
        val_files = [ [ files_main_u[key]['X'],files_main_u[key]['y']] for key in files_main_u.keys() if key in valList]

        if single_user:
            train_x, train_y = merge_files(train_files)  # Single user
        else:
            train_x, train_y = merge_files(train_files+files_other_u) #Multiuser

        val_x, val_y = merge_files(val_files)

        #Normalize
        # train_x = np.delete(train_x,3,axis=1)
        # val_x = np.delete(val_x, 3, axis=1)
        global_mean = train_x.mean(axis=0).reshape(1,number_of_feat)
        global_std = train_x.std(axis=0).reshape(1,number_of_feat)
        train_x = (train_x - global_mean) / global_std
        val_x = (val_x - global_mean) / global_std

        #Create model
        model = create_baseline(number_of_feat)
        #Train
        history = model.fit(train_x, train_y,
                        batch_size=20,
                        epochs=120,
                        validation_data=(val_x,val_y),
                        shuffle=True,
                        verbose=False)

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

        plotTitle = "eye tracker"
        plt.plot(history.history['accuracy'], label="train")
        plt.plot(history.history['val_accuracy'], label='val')
        plt.legend()
        # plt.show()
        plt.close()

    ##############################
    # Train a model with all data#
    ##############################
    # train_files1 = [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys()]
    #
    # if single_user:
    #     print("train single user model")
    #     train_x, train_y = merge_files(train_files1)  # Single user
    # else:
    #     train_x, train_y = merge_files(train_files1 + files_other_u)  # Multiuser
    #
    # # Normalize
    # # train_x = np.delete(train_x, 3, axis=1)
    # global_mean = train_x.mean(axis=0).reshape(1, number_of_feat)
    # global_std = train_x.std(axis=0).reshape(1, number_of_feat)
    # train_x = (train_x - global_mean) / global_std
    # # Create model
    # print("train shape", train_x.shape)
    # model = create_baseline(number_of_feat)
    # # Train
    # history = model.fit(train_x, train_y,
    #                     batch_size=20,
    #                     epochs=120,
    #                     shuffle=True,
    #                     verbose=True)
    #
    # model.save('./model/model_{:}_fuse.h5'.format(model_name))
    # normalizer = {'mean':global_mean,'std':global_std}
    # pickle.dump(normalizer, open('./model/normalizer_{:}_fuse.pic'.format(model_name),'wb'))

if __name__ == "__main__":
    main()