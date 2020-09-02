"""
Test transfer learning across users
"""

from PowerClassification.Utils.NetworkTraining import CrossValidationRoutines
from PowerClassification.Utils.NetworkTraining import Utils
from PowerClassification.Utils.NetworkTraining import TransferLearningModule
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

['ryan','juan', 'jackie','jhony', 'karuna','santy']
'''

if __name__ == '__main__':
    USERS = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']
    #EEG_CHANNELS = ['FZ', 'F7', 'F3', 'F4', 'F8']
    EEG_CHANNELS = ["FP1", "FP2", "F7", "F3", "FZ", "F4",
                     "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
                     "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
                     "PZ", "P4", "P8", "PO3", "PO4", "OZ"]
    POWER_COEFFICIENTS = ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    LSTM_SAMPLE_SIZE = [140]
    WINDOW_SIZE = [10]
    RESULTS_ROOT = Path('./results/results_transfer10/') / 'aa11_pyprep/'
    rootPath = Path('./data/de-identified-pyprep-dataset-reduced/')

    transferModule = TransferLearningModule()
    utilsModule = Utils()

    #Write experiment parameters
    params = {'users':USERS,'eeg channels':EEG_CHANNELS,
              'power coefficients':POWER_COEFFICIENTS,
              'lstm sample sizes':LSTM_SAMPLE_SIZE,
              'window size': WINDOW_SIZE,
              'results root':RESULTS_ROOT}

    for lstmSampleSize,windowSize in itertools.product(LSTM_SAMPLE_SIZE,WINDOW_SIZE):

        lstmSteps = int(lstmSampleSize/windowSize)
        completeResults = []


        for user in USERS:
            dataPath = rootPath / '{:02d}s/'.format(windowSize)
            resultsPath = RESULTS_ROOT / 'window{:02d}s_sampleSize{:02d}s'.format(windowSize,lstmSampleSize)

            if not resultsPath.exists():
                print('create ', resultsPath)
                Path.mkdir(resultsPath,parents=True)

            results = transferModule.transferCrossValidation(lstmSteps, dataPath,resultsPath,
                                                             user,USERS,eegChannels=EEG_CHANNELS,
                                                             powerCoefficients=POWER_COEFFICIENTS,
                                                             dataFormat='time')

            results.to_csv(resultsPath / '{:}_results.csv'.format(user), index= False)
            completeResults.append(results)

        completeResults = pd.concat(completeResults)
        completeResults.to_csv(resultsPath.parent / 'window{:02d}s_sampleSize{:02d}s.csv'.format(windowSize,lstmSampleSize) , index = False)
        #utilsModule.sendSMS("Finished window{:02d}s_sampleSize{:02d}s".format(windowSize,lstmSampleSize))

    with open(resultsPath.parent / 'experimentParams.txt','w') as f:
        for k,i in params.items():
            line = str(k)+':'+str(i)+'\n'
            f.write(line)
