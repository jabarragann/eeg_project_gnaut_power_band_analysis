from pathlib import Path
from EyeTrackerClassification.ClassifyFuseFeatures import cv_routine, load_files
import pandas as pd

list_of_path = \
[ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UPaul\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJuan\S11"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\URunping\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments4-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S1"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S6"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UKeyu\S1"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UNan\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UArturo\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChenxi\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S01"),
 Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAndres\S01"),
  ]


# list_of_path = \
# [ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UNan\S01"),
#   Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChiho\S01"),
#   Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UArturo\S01"),
#   Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UChenxi\S01"),
#   Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UAlfredo\S01"),]

list_of_path = \
[ Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UBen\S01"),]

def main():
    results = []
    for p in list_of_path:
        print("Training model")
        print(p.exists())

        model_name = 'all-users-9-feat'
        single_user = True
        path_main_u = p

        labels = ['NeedlePassing', 'BloodNeedle']
        files_main_u = load_files(path_main_u,labels)

        session = path_main_u.name
        user = path_main_u.parent.name
        results_df = cv_routine(files_main_u,None,single_user=single_user,user=user,session=session)

        results.append(results_df)

    results = pd.concat(results)
    results.to_csv('./results/cv_individual.csv')

if __name__ == "__main__":
    main()
