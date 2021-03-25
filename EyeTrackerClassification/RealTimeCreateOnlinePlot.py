import sys
sys.path.append('./../')
import pickle
from pathlib import Path
import EyeTrackerClassification.EyeTrackerUtils as etu
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def main():
    path_plot = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJuan\S8-Validation-2')
    path_mouse_e = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\eyetracker\Juan\S8-validation-2')
    labels = ['NeedlePassing', 'BloodNeedle']

    predictions = pd.read_csv(path_mouse_e / "PredictionEvents_data.csv")

    start_time = predictions.loc[0,"0"]

    mouse_file = path_mouse_e / 'MouseButtons_data.csv'
    df2 = pd.read_csv(mouse_file)
    for i in range(df2.shape[0]):
        event = df2.loc[i, '1']
        if event == 'MouseButtonX2 pressed':
            plt.axvline(df2.loc[i, '0'] - start_time, color='red')

    user_ses = path_plot.parent.name + "/" + path_plot.name
    plt.title("online predictions from data: {:}".format(user_ses))
    plt.plot(predictions.loc[:,"0"]-start_time,predictions.loc[:,"1"], '*-', markersize=4,)
    plt.show()

if __name__ =='__main__':
    main()