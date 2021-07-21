from pathlib import Path
from EyeTrackerClassification.ClassifyFuseFeatures import cv_routine, load_files, merge_files, train_test
import pandas as pd
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from sklearn.metrics import roc_curve, auc
import numpy as np

########## Model selection ##################################
multi_user_path = \
[ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UPaul\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJuan\S11"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\URunping\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UKeyu\S1"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UNan\S01"),
 # Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChiho\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UArturo\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChenxi\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S01"),
 # Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UGaojian\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAndres\S01"),
  ]

## Idea 1 --> multi + S6 --> S7-S2
list_of_path = \
[ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6")]
list_of_path = list_of_path + multi_user_path
test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S07-S2")
model_type='big'
model_name='idea1'

threshold_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6-02")

## Idea 2
## Add transfer to 'idea1' model

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

def main():
    results = []
    data = []

    labels = ['NeedlePassing', 'BloodNeedle']
    for p in list_of_path:
        print("Training model")
        print(p.exists())

        # model_name = 'all-users-9-feat'
        # single_user = True
        path_main_u = p

        files_main_u = load_files(path_main_u,labels)
        data += [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys()]


    #Load training data
    train_x, train_y = merge_files(data)
    # train_x = train_x[:,:-4]

    #Load testing data
    test_data = load_files(test_user_path, labels)
    test_data = [[test_data[key]['X'], test_data[key]['y']] for key in test_data.keys()]
    test_x, test_y = merge_files(test_data)
    # test_x = test_x[:, :-4]

    #train classifier
    model = train_test(train_x,train_y,test_x,test_y, number_of_feat=train_x.shape[1],model_name_2=model_name,model_type=model_type)

    #Get threshold
    threshold_data = load_files(threshold_path, labels)
    threshold_data = [[threshold_data[key]['X'], threshold_data[key]['y']] for key in threshold_data.keys()]
    thresh_x, thresh_y = merge_files(threshold_data)

    predicted = model.predict(thresh_x)
    optimal_thresh = Find_Optimal_Cutoff(thresh_y, predicted)
    print(optimal_thresh)

if __name__ == "__main__":
    main()
