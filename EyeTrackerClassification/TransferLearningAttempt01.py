import pickle
from pathlib import Path
from EyeTrackerClassification.ClassifyFuseFeatures import cv_routine, load_files, merge_files, train_test
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def params_count(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))

def get_data(data_path):
    labels = ['NeedlePassing', 'BloodNeedle']
    test_data = load_files(data_path, labels)
    test_data = [[test_data[key]['X'], test_data[key]['y']] for key in test_data.keys()]
    x,y = merge_files(test_data)
    return x,y

def test_model(model, test_x, test_y):
    results = model.evaluate(test_x, test_y,verbose=False)
    predictions = model.predict(test_x)
    predicted_label = predictions.copy()
    threshold = 0.50
    predicted_label[predicted_label > threshold] = 1.0
    predicted_label[predicted_label < threshold] = 0.0
    print('ACC {:.04f}'.format(results[1]))
    tn, fp, fn, tp = confusion_matrix(test_y, predicted_label).ravel()
    print("tn, fp, fn,tp", tn, fp, fn, tp)

def create_baseline(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    model_name = "multiuser01-alfredo-real-time-exp"
    transfer_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S03")
    test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S02")

    #Load testing data
    trans_x, trans_y = get_data(transfer_path)
    test_x, test_y = get_data(test_user_path)

    #Load model
    model_path = './model/model_{:}_fuse.h5'.format(model_name)
    model = keras.models.load_model(model_path)
    normalizer = pickle.load( open('./model/normalizer_{:}_fuse.pic'.format(model_name), 'rb'))

    #Normalize features
    global_mean = normalizer['mean']
    global_std = normalizer['std']
    test_x = (test_x - global_mean) / global_std
    trans_x = (trans_x - global_mean) / global_std

    # Predict model - before transfer
    test_model(model, test_x, test_y)

    #Fine freeze every layer except last ones.
    for l in model.layers[:-20]:#20 seems to be the magical number
        l.trainable = False

    opti = keras.optimizers.Adam(learning_rate=0.001)
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
    # opti = keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])

    # model.summary()
    params_count(model)

    #Fine tune model
    history = model.fit(trans_x, trans_y,
                        batch_size=10,
                        epochs=150,
                        shuffle=True,
                        verbose=False)
    print("Model after transfer learning")
    test_model(model, test_x, test_y)

    plt.plot(history.history['accuracy'], label="train")
    plt.legend()
    plt.show()
    plt.close()

    model.save('./model/model_{:}_fuse.h5'.format(model_name+"-dani-transfer"))

    #Test naive model
    # naive_model = create_baseline(9)
    # history = model.fit(trans_x, trans_y,
    #                     batch_size=10,
    #                     epochs=150,
    #                     shuffle=True,
    #                     verbose=False)
    # print("Naive model performance")
    # test_model(naive_model,test_x, test_y)

if __name__ == "__main__":
    main()
