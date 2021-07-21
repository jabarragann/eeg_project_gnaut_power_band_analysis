import sys
sys.path.append('./../')
import pickle
from pathlib import Path
import EyeTrackerClassification.EyeTrackerUtils as etu
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def main():
    path_plot = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UDani\S03')
    path_mouse_e = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\eyetracker\Dani\S03')
    labels = ['NeedlePassing', 'BloodNeedle']

    model_name='individual01-dani-real-time-exp'
    predictions, ts = pickle.load(open(path_plot / 'pred_plot_{:}_fuse.pic'.format(model_name),'rb'))
    start_time = ts[0]

    mouse_file = path_mouse_e / 'MouseButtons_data.csv'
    df2 = pd.read_csv(mouse_file)
    for i in range(df2.shape[0]):
        event = df2.loc[i, '1']
        if event == 'MouseButtonX2 pressed':
            plt.axvline(df2.loc[i, '0'] - start_time, color='red')

    #Smoothed curve
    predictions_smooth = pd.DataFrame(predictions,columns=['predictions'])
    predictions_smooth = predictions_smooth.rolling(12).mean()

    user_ses = path_plot.parent.name + "/" + path_plot.name
    plt.title("data: {:} model: {:}".format(user_ses,model_name))
    plt.plot(ts-start_time,predictions, '*-')
    plt.plot(ts-start_time,predictions_smooth['predictions'])
    plt.grid()
    plt.ylabel('workload index - EEG - Eye tracker')
    plt.xlabel('time (s)')
    plt.show()

if __name__ =='__main__':
    main()