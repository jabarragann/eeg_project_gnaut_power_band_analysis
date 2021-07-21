from pathlib import Path
from EyeTrackerClassification.ClassifyFuseFeatures import load_files, merge_files

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


if __name__ == "__main__":
    results = []
    data = []
    for p in multi_user_path:
        print("Training model")
        print(p.exists())

        path_main_u = p

        labels = ['NeedlePassing', 'BloodNeedle']
        files_main_u = load_files(path_main_u, labels)
        data += [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys()]

    # Load training data
    train_x, train_y = merge_files(data)


    x = 0