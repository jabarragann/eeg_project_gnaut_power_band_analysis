'''
Grid search for the best lstm sample size and windows size.

multi user
'''

from PowerClassification.Utils.NetworkTraining import CrossValidationRoutines
from PowerClassification.Utils.NetworkTraining import Utils
from pathlib import Path
import pandas as pd
import itertools

'''
EEG_CHANNELS = [
        "FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
        "PZ", "P4", "P8", "PO3", "PO4", "OZ"]
POWER_COEFFICIENTS = ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
'''

if __name__ == '__main__':
    USERS = ['ryan','juan', 'jackie','jhony']
    EEG_CHANNELS = ['FZ', 'F7', 'F3', 'F4', 'F8']
    POWER_COEFFICIENTS = ['Theta', 'Alpha', 'Beta', 'Gamma']
    LSTM_SAMPLE_SIZE = [120]
    WINDOW_SIZE = [10,15,20]

    crossValidationModule = CrossValidationRoutines()
    utilsModule = Utils()

    #Write experiment parameters
    params = {'users':USERS,'eeg channels':EEG_CHANNELS,
              'power coefficients':POWER_COEFFICIENTS,
              'lstm sample sizes':LSTM_SAMPLE_SIZE,
              'window size': WINDOW_SIZE}

    for lstmSampleSize,windowSize in itertools.product(LSTM_SAMPLE_SIZE,WINDOW_SIZE):

        lstmSteps = int(lstmSampleSize/windowSize)
        completeResults = []

        for user in ['ryan','juan', 'jackie','jhony']:
            dataPath = Path('./data/DifferentWindowSizeData/{:02d}s/'.format(windowSize))
            resultsPath = Path('./results/results_transfer9') / 'selecCoef/f-TABG-coeffExp1'/ 'window{:02d}s_sampleSize{:02d}s'.format(windowSize,lstmSampleSize)

            if not resultsPath.exists():
                print('create ', resultsPath)
                Path.mkdir(resultsPath,parents=True)

            results = crossValidationModule.userCrossValidationMultiUser(lstmSteps, dataPath,resultsPath,
                                                                         user,USERS,eegChannels=EEG_CHANNELS,
                                                                         powerCoefficients=POWER_COEFFICIENTS)
            results.to_csv(resultsPath / '{:}_results.csv'.format(user), index= False)
            completeResults.append(results)

        completeResults = pd.concat(completeResults)
        completeResults.to_csv(resultsPath.parent / 'window{:02d}s_sampleSize{:02d}s.csv'.format(windowSize,lstmSampleSize) , index = False)
        utilsModule.sendSMS("Finished window{:02d}s_sampleSize{:02d}s".format(windowSize,lstmSampleSize))

    with open(resultsPath.parent / 'experimentParams.txt','w') as f:
        for k,i in params.items():
            line = str(k)+':'+str(i)+'\n'
            f.write(line)
