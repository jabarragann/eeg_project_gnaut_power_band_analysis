from pathlib import Path
from EyeTrackerClassification.ClassifyFuseFeatures import cv_routine, load_files
import pandas as pd
from EyeTrackerClassification.ClassifyFuseFeatures import cv_routine, load_files, merge_files, train_test, train_test_without_saving_model

def main():
    root_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Realtime-Project-Purdue-experiments\fusefeatures")
    model_type = 'small'
    labels = ['NeedlePassing', 'BloodNeedle']
    multi_user_path =  [p for p in  root_path.glob("*/S01") if p.parent.name not in ['UArturo','UGaojian','UBen','UChiho']]
    results_df = pd.DataFrame(columns=['user', 'test_acc'])

    for test_data_path in multi_user_path:
        print("Testing pathmodel:")
        print(test_data_path.exists(), test_data_path.parent.name)
        train_data = multi_user_path.copy()
        train_data.remove(test_data_path)
        x=0
        # Load train data
        data = []
        for p in train_data:
            path_main_u = p
            labels = ['NeedlePassing', 'BloodNeedle']
            files_main_u = load_files(path_main_u, labels)
            data += [[files_main_u[key]['X'], files_main_u[key]['y']] for key in files_main_u.keys()]
        train_x, train_y = merge_files(data)

        # Load testing data
        test_data = load_files(test_data_path, labels)
        test_data = [[test_data[key]['X'], test_data[key]['y']] for key in test_data.keys()]
        test_x, test_y = merge_files(test_data)

        # train classifier
        results = train_test_without_saving_model(train_x, train_y, test_x, test_y, number_of_feat=train_x.shape[1],
                                                  model_type=model_type)
        print(results)

        results_dict = {'user': test_data_path.parent.name, 'test_acc': results['test_acc']}
        results_df = results_df.append(results_dict, ignore_index=True)
        results_df.to_csv('./results/multi_user_test.csv')


if __name__ == "__main__":
    main()
