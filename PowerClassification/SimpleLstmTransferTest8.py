'''
Grid search for the best lstm sample size and windows size.
'''

from PowerClassification.Utils.NetworkTraining import CrossValidationRoutines
from PowerClassification.Utils.NetworkTraining import Utils
from pathlib import Path
import pandas as pd
import itertools

if __name__ == '__main__':
    USERS = ['ryan','juan', 'jackie','jhony']
    LSTM_SAMPLE_SIZE = [30,45,60,75,90]
    WINDOW_SIZE = [5,10,15,20,25,30]

    crossValidationModule = CrossValidationRoutines()
    utilsModule = Utils()

    for lstmSampleSize,windowSize in itertools.product(LSTM_SAMPLE_SIZE,WINDOW_SIZE):

        lstmSteps = int(lstmSampleSize/windowSize)
        completeResults = []

        for user in ['ryan','juan', 'jackie','jhony']:
            userDataPath = Path('./data/DifferentWindowSizeData/{:02d}s/{:}/'.format(windowSize, user))
            resultsPath = Path('./results/results_transfer8') / 'window{:02d}s_sampleSize{:02d}s'.format(windowSize,lstmSampleSize)

            if not resultsPath.exists():
                print('create ', resultsPath)
                Path.mkdir(resultsPath,parents=True)

            results = crossValidationModule.userCrossValidation(lstmSteps, userDataPath,resultsPath, user)
            results.to_csv(resultsPath / '{:}_results.csv'.format(user), index= False)
            completeResults.append(results)

        completeResults = pd.concat(completeResults)
        completeResults.to_csv(resultsPath.parent / 'window{:02d}s_sampleSize{:02d}s.csv'.format(windowSize,lstmSampleSize) , index = False)
        utilsModule.sendSMS("Finished window{:02d}s_sampleSize{:02d}s".format(windowSize,lstmSampleSize))
