import pickle
from pathlib import Path
import EyeTrackerClassification.EyeTrackerUtils as etu
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def main():
    path = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\eyetracker\Jing\S3-validation')
    labels = ['NeedlePassing', 'BloodNeedle']

    user='jing'
    predictions, ts = pickle.load(open('./real_time_pred/pred_plot_{:}_fuse.pic'.format(user),'rb'))
    start_time = ts[0]

    mouse_file = path / 'MouseButtons_data.csv'
    df2 = pd.read_csv(mouse_file)
    for i in range(df2.shape[0]):
        event = df2.loc[i, '1']
        if event == 'MouseButtonX2 pressed':
            plt.axvline(df2.loc[i, '0'] - start_time, color='red')

    plt.title("{:} validation data - eye tracker".format(user))
    plt.plot(ts-start_time,predictions, '*-')
    plt.show()

if __name__ =='__main__':
    main()