import pickle
from pathlib import Path
import EyeTrackerClassification.EyeTrackerUtils as etu
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def main():
    path = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\eyetracker\S3-validation')
    labels = ['NeedlePassing', 'BloodNeedle']

    files_dict = etu.search_files_on_path(path)
    files_dict = etu.merge_files(files_dict, labels)

    # Load model & normalizer
    model = load_model('./model/model.h5')
    normalizer = pickle.load(open('./model/normalizer.pic','rb'))
    global_mean = normalizer['mean']
    global_std = normalizer['std']

    #Load data
    train_files = [files_dict[key] for key in files_dict.keys()]
    test_x, test_y,ts_events = etu.get_data_single_file(train_files[0])
    test_x = test_x.values
    test_y = test_y.values

    #Normalize data
    test_x = (test_x - global_mean) / global_std

    #Predict
    predictions = model.predict(test_x)

    #Save predictions
    pickle.dump([predictions,ts_events],open('./real_time_pred/pred_plot.pic','wb'))

    #Plot results
    plt.plot(predictions)
    plt.show()

if __name__ == "__main__":
    main()