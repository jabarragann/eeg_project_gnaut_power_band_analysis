'''
Grid search for the best lstm sample size and windows size.

multi user
'''

from PowerClassification.Utils.NetworkTraining import CrossValidationRoutines
from PowerClassification.Utils.NetworkTraining import Utils
from pathlib import Path
import pandas as pd
import itertools

if __name__ == '__main__':
    USERS = ['ryan','juan', 'jackie','jhony']
    EEG_CHANNELS = ['FZ', 'F7', 'F3', 'F4', 'F8']
    LSTM_SAMPLE_SIZE = [60,75,90,105,120]
    WINDOW_SIZE = [10,15,20,25,30,40]

    crossValidationModule = CrossValidationRoutines()
    utilsModule = Utils()

    for lstmSampleSize,windowSize in itertools.product(LSTM_SAMPLE_SIZE,WINDOW_SIZE):

        lstmSteps = int(lstmSampleSize/windowSize)
        completeResults = []

        for user in ['ryan','juan', 'jackie','jhony']:
            dataPath = Path('./data/DifferentWindowSizeData/{:02d}s/'.format(windowSize))
            resultsPath = Path('./results/results_transfer9') / 'fChannelsExp-NoICA'/ 'window{:02d}s_sampleSize{:02d}s'.format(windowSize,lstmSampleSize)

            if not resultsPath.exists():
                print('create ', resultsPath)
                Path.mkdir(resultsPath,parents=True)

            results = crossValidationModule.userCrossValidationMultiUser(lstmSteps, dataPath,resultsPath, user,USERS,eegChannels=EEG_CHANNELS)
            results.to_csv(resultsPath / '{:}_results.csv'.format(user), index= False)
            completeResults.append(results)

        completeResults = pd.concat(completeResults)
        completeResults.to_csv(resultsPath.parent / 'window{:02d}s_sampleSize{:02d}s.csv'.format(windowSize,lstmSampleSize) , index = False)
        utilsModule.sendSMS("Finished window{:02d}s_sampleSize{:02d}s".format(windowSize,lstmSampleSize))
