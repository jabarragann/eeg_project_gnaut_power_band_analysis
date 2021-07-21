from pathlib import Path
from EyeTrackerClassification.ClassifyFuseFeatures import cv_routine, load_files, merge_files, train_test, train_test_without_saving_model
import pandas as pd

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)



# list_of_path = \
# [ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UPaul\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJuan\S11"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\URunping\S01"),
#  # Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UKeyu\S1"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UNan\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChiho\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UArturo\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChenxi\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UGaojian\S01"),
#  Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAndres\S01"),
#   ]
#
# test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1")

#########################################################
###### Model selection ##################################
multi_user_path = \
[ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UPaul\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\URunping\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UKeyu\S1"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UNan\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UArturo\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChenxi\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAndres\S01"),
  ]

## Idea 1 --> multi + S6 --> S7-S2
# list_of_path = \
# [ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1")]
list_of_path =  multi_user_path  #+ list_of_path
test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S1")
model_type='big'
model_name='multiuser-to-erase'

## Idea 2
## Add transfer to 'idea1' model

# # ## Idea 3
# list_of_path = \
# [ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6")]
# test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S07-S2")
# model_type='small'
# model_name='idea3'

# Idea 4
# list_of_path = \
# [ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAndres\S01"),]
# test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAndres\S01")
# model_type='small'
# model_name='individual01-andres-real-time-exp'

## Idea 5
# list_of_path = multi_user_path
# test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S07-S2")
# model_type='big'
# model_name='idea5'


# list_of_path = \
# [ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S2"),]
# test_user_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6")


def main():
    results = []
    data = []
    for p in list_of_path:
        print("Training model")
        print(p.exists())

        path_main_u = p

        labels = ['NeedlePassing', 'BloodNeedle']
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

def generate_multi_u_results():
    results = []
    data = []
    results_df = pd.DataFrame(columns=['user','test_acc'])
    for test_data_path in multi_user_path:
        print("Testing pathmodel")
        print(test_data_path.exists(), test_data_path.parent.name)
        train_data = multi_user_path.copy()
        train_data.remove(test_data_path)

        #Load train data
        for p in train_data:
            path_main_u = p
            labels = ['NeedlePassing', 'BloodNeedle']
            files_main_u = load_files(path_main_u,labels)
            data += [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys()]
        train_x, train_y = merge_files(data)

        #Load testing data
        test_data = load_files(test_data_path, labels)
        test_data = [[test_data[key]['X'], test_data[key]['y']] for key in test_data.keys()]
        test_x, test_y = merge_files(test_data)

        #train classifier
        results = train_test_without_saving_model(train_x,train_y,test_x,test_y, number_of_feat=train_x.shape[1],model_type=model_type)
        print(results)

        results_dict = {'user':test_data_path.parent.name,'test_acc':results['test_acc']}
        results_df = results_df.append(results_dict,ignore_index=True)
        results_df.to_csv('./results/multi_user_test.csv')

if __name__ == "__main__":
    #main()
    generate_multi_u_results()
