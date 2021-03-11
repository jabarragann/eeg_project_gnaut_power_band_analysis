import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":


    # file = "./CheckPredictionPlot/UJuan_S01_T01_BloodValidation_raw_model_predictions.csv"
    file = "./CheckPredictionPlot/UJuan_S01_T01_EyesValidation_raw_model_predictions.csv"
    df = pd.read_csv(file,index_col=[0])
    start_time = df["LSL_TIME"].values[0]


    # mouse_file = './CheckPredictionPlot/UJuan_S01_T01_BloodValidation_raw_MouseButtons_data.csv'
    # df2 = pd.read_csv(mouse_file)
    # for i in range(df2.shape[0]):
    #     event = df2.loc[i, '1']
    #     if event == 'MouseButtonX2 pressed':
    #         plt.axvline(df2.loc[i, '0'] - start_time, color='red')

    plt.plot(df["LSL_TIME"]- start_time,df["MODEL_PREDICTION"])
    plt.show()

