import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
import EyeTrackerClassification.EyeTrackerUtils as etu
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve, auc, confusion_matrix

import tensorflow as tf

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    fpr,tpr, threshold = fpr[1:], tpr[1:], threshold[1:]
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - fpr, index=i), 'threshold': pd.Series(threshold, index=i)})
    roc = roc.loc[roc['threshold']>0.5,:]
    roc_t = roc.iloc[(roc.tf - 0).argsort()[:4]]

    return list(roc_t['threshold'])

def create_roc_curve(testY, y_score):
    fpr, tpr, thresh = roc_curve(testY, y_score)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()

def create_baseline(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_2(input_dim):
    # create model
    model_input  = Input(shape=input_dim)
    # x = Dropout(0.01)(model_input)
    x = Dense(30, input_dim=input_dim, activation='relu')(model_input)
    x = BatchNormalization()(x)

    for i in range(8): #8
        x = Dropout(0.05,)(x)
        x = Dense(25, input_dim=input_dim, activation='relu')(x) #,kernel_regularizer=regularizers.l2(0.0001)
        x = BatchNormalization()(x)

    model_output = Dense(1, activation='sigmoid')(x)

    model = Model(model_input, model_output)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

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

def train_test(train_x, train_y, val_x,val_y, number_of_feat=9, model_name_2="final_final_all_users", model_type='big'):

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

    global_mean = train_x.mean(axis=0).reshape(1, number_of_feat)
    global_std = train_x.std(axis=0).reshape(1, number_of_feat)
    train_x = (train_x - global_mean) / global_std
    val_x = (val_x - global_mean) / global_std

    print("train shape", train_x.shape)
    print("val shape", val_x.shape)

    # Create model
    if model_type == 'big':
        model = model_2(number_of_feat)
    elif model_type == 'small':
        model = create_baseline(number_of_feat)
    else:
        raise Exception("Error")

    model.summary()
    # Train
    history = model.fit(train_x, train_y,
                        batch_size=10,
                        epochs=100,
                        validation_data=(val_x, val_y),
                        shuffle=True,
                        verbose=True,
                        callbacks=[callback],)

    predicted = model.predict(val_x)
    predicted_label = predicted.copy()
    threshold = 0.50
    predicted_label[predicted > threshold] = 1.0
    predicted_label[predicted < threshold] = 0.0
    acc1 = sum(val_y.squeeze() == predicted_label.squeeze()) / val_y.shape[0]
    print("acc with new threshold", acc1)

    plotTitle = "eye tracker"
    plt.plot(history.history['accuracy'], label="train")
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.show()
    plt.close()

    print(model_name_2)
    model.save('./model/model_{:}_fuse.h5'.format(model_name_2))
    normalizer = {'mean': global_mean, 'std': global_std}
    pickle.dump(normalizer, open('./model/normalizer_{:}_fuse.pic'.format(model_name_2), 'wb'))

    return model

def train_test_without_saving_model(train_x, train_y, val_x,val_y, number_of_feat=9,  model_type='big'):
    """
    Not the best way to test. I am doing early stopping in my testing set!
    :param train_x:
    :param train_y:
    :param val_x:
    :param val_y:
    :param number_of_feat:
    :param model_type:
    :return:
    """
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

    global_mean = train_x.mean(axis=0).reshape(1, number_of_feat)
    global_std = train_x.std(axis=0).reshape(1, number_of_feat)
    train_x = (train_x - global_mean) / global_std
    val_x = (val_x - global_mean) / global_std

    print("train shape", train_x.shape)
    print("val shape", val_x.shape)

    # Create model
    if model_type == 'big':
        model = model_2(number_of_feat)
    elif model_type == 'small':
        model = create_baseline(number_of_feat)
    else:
        raise Exception("Error")

    model.summary()
    # Train
    history = model.fit(train_x, train_y,
                        batch_size=10,
                        epochs=100,
                        validation_data=(val_x, val_y),
                        shuffle=True,
                        verbose=True,
                        callbacks=[callback],)

    predicted = model.predict(val_x)
    predicted_label = predicted.copy()
    threshold = 0.50
    predicted_label[predicted > threshold] = 1.0
    predicted_label[predicted < threshold] = 0.0
    val_acc= sum(val_y.squeeze() == predicted_label.squeeze()) / val_y.shape[0]
    print("acc with new threshold", val_acc)

    #plotTitle = "eye tracker"
    #plt.plot(history.history['accuracy'], label="train")
    #plt.plot(history.history['val_accuracy'], label='val')
    #plt.legend()
    #plt.show()
    #plt.close()

    return {'test_acc':val_acc}

def cv_routine(files_main_u, files_other_u, single_user,user,session):
    number_of_feat = 9

    results_df = pd.DataFrame(columns=['user','session','cv','train_shape','val_shape','val_acc'])

    count = 0
    # split into train val
    for valList in [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]:
        train_files = [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys() if key not in valList]
        val_files = [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys() if key in valList]

        if len(val_files) == 0:
            print("skipping ", valList)
            continue

        if single_user:
            train_x, train_y = merge_files(train_files)  # Single user
        else:
            train_x, train_y = merge_files(train_files + files_other_u)  # Multiuser

        val_x, val_y = merge_files(val_files)

        # Normalize
        # train_x = np.delete(train_x,3,axis=1)
        # val_x = np.delete(val_x, 3, axis=1)
        train_x = train_x
        global_mean = train_x.mean(axis=0).reshape(1, number_of_feat)
        global_std = train_x.std(axis=0).reshape(1, number_of_feat)
        train_x = (train_x - global_mean) / global_std
        val_x = (val_x - global_mean) / global_std

        # Create model
        model = create_baseline(number_of_feat)
        # Train
        history = model.fit(train_x, train_y,
                            batch_size=20,
                            epochs=100,
                            validation_data=(val_x, val_y),
                            shuffle=True,
                            verbose=False)

        # predict
        predicted = model.predict(val_x)
        predicted_label = predicted.copy()
        threshold = 0.50
        predicted_label[predicted > threshold] = 1.0
        predicted_label[predicted < threshold] = 0.0
        acc1 = sum(val_y.squeeze() == predicted_label.squeeze()) / val_y.shape[0]
        acc2 = model.evaluate(val_x,val_y,verbose=0)[1]
        optimal_thresh = Find_Optimal_Cutoff(val_y, predicted)
        # create_roc_curve(val_y,predicted)

        # print("optimal threshold", optimal_thresh)
        print("Shapes: ", train_x.shape, val_x.shape)
        print(valList, "nn_acc", "threshold {:.3f}".format(threshold))
        print("{:0.3f}".format(acc1))

        # ['user', 'session', 'cv', 'train_shape', 'val_shape', 'train_acc', 'val_acc']
        results_df.loc[count, ['user','session','cv']] = user,session,str(valList)
        results_df.loc[count, ['train_shape','val_shape']] = str(train_x.shape), str(val_x.shape)
        results_df.loc[count, ['val_acc']] = acc1

        tn, fp, fn, tp = confusion_matrix(val_y, predicted_label).ravel()
        s = val_x.shape[0]
        print("tn {:d} fp {:d} fn {:d} tp {:d} total {:d}".format(tn, fp, fn, tp ,s))
        # print("tn {:0.3f} fp {:0.3f} fn {:0.3f} tp {:0.3f}".format(tn/s, fp/s, fn/s, tp/s))
        count += 1 ## DON'T COMMENT THIS LINE!

        #########Plot Functions #############
        # plotTitle = "eye tracker"
        # plt.plot(history.history['accuracy'], label="train")
        # plt.plot(history.history['val_accuracy'], label='val')
        # plt.legend()
        # plt.show()
        # plt.close()
    return results_df

def main():
    model_name = 'all-users-9-feat'
    single_user = False
    number_of_feat = 9
    path_main_u = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1")

    # path2 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJuan\S7")
    # path2 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UKeyu\S1")
    # path3 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1")
    # path4 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6")
    # path5 = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1")

    labels = ['NeedlePassing', 'BloodNeedle']
    files_main_u = load_files(path_main_u,labels)

    session = path_main_u.name
    user = path_main_u.parent.name
    results_df = cv_routine(files_main_u,None,single_user=True,user=user,session=session)

    x = 0

    # files_dict2 = load_files(path2,labels)
    # files2 = [[files_dict2[key]['X'], files_dict2[key]['y']] for key in files_dict2.keys()]
    # files_dict3 = load_files(path3, labels)
    # files3 = [[files_dict3[key]['X'], files_dict3[key]['y']] for key in files_dict3.keys()]
    # files_dict4 = load_files(path4, labels)
    # files4 = [[files_dict4[key]['X'], files_dict4[key]['y']] for key in files_dict4.keys()]
    # # files_dict5 = load_files(path5, labels)
    # # files5 = [[files_dict5[key]['X'], files_dict5[key]['y']] for key in files_dict5.keys()]
    # files_other_u = files2+files3+files4 #+files5



    ##############################
    # Train a model with all data#
    ##############################
    # train_files1 = [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys()]
    #
    # if single_user:
    #     print("train single user model")
    #     train_x, train_y = merge_files(train_files1)  # Single user
    # else:
    #     print("train multi user model")
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