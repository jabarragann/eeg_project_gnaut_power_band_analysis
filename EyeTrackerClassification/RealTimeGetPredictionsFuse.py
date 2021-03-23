import pickle
from pathlib import Path
import EyeTrackerClassification.EyeTrackerUtils as etu
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import re
def load_files(path,labels):
    files_dict = {}
    for file in path.rglob("*.txt"):
        task = re.findall('(?<=_S[0-9]{2}_T[0-9]{2}_).+(?=_fuse)', file.name)[0]
        trial = int(re.findall('(?<=_S[0-9]{2}_T)[0-9]{2}(?=_)', file.name)[0])
        label = 1.0 if labels[0]==task else 0.0
        print(file.name,task,label)
        df = pd.read_csv(file,index_col=[0])
        files_dict[trial] = {'X':df}

    return files_dict

def main():
    path = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\fusefeatures\UJing\S03-validation')
    labels = ['NeedlePassing', 'BloodNeedle']

    files_dict = load_files(path, labels)


    # Load model & normalizer
    user = 'jing'
    model = load_model('./model/model_{:}_fuse.h5'.format(user))
    normalizer = pickle.load(open('./model/normalizer_{:}_fuse.pic'.format(user),'rb'))
    global_mean = normalizer['mean']
    global_std = normalizer['std']

    #Load data
    train_files = [files_dict[key] for key in files_dict.keys()]
    test_x =  train_files[0]['X']
    ts_events = test_x['LSL_TIME'].values
    test_x  = test_x.drop('LSL_TIME', axis=1).values

    #Normalize data
    test_x = (test_x - global_mean) / global_std

    #Predict
    predictions = model.predict(test_x)

    #Save predictions
    pickle.dump([predictions,ts_events],open('./real_time_pred/pred_plot_{:}_fuse.pic'.format(user),'wb'))

    #Plot results
    plt.plot(predictions)
    plt.show()

if __name__ == "__main__":
    main()