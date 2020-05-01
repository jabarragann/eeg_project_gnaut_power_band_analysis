'''
Grid search for the best lstm sample size and windows size.

multi user
'''

from PowerClassification.Utils.NetworkTraining import CrossValidationRoutines
from PowerClassification.Utils.NetworkTraining import Utils
from pathlib import Path
import pandas as pd
import itertools
import argparse
from types import SimpleNamespace

'''
EEG_CHANNELS = [
        "FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
        "PZ", "P4", "P8", "PO3", "PO4", "OZ"]
POWER_COEFFICIENTS = ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
'''

def readCommandLineArg(params):

	parser = argparse.ArgumentParser()
	parser.add_argument('--TESTED_USERS')
	parser.add_argument('--ALL_USERS')
	parser.add_argument('--EEG_CHANNELS')
	parser.add_argument('--POWER_COEFFICIENTS')
	parser.add_argument('--LSTM_SAMPLE_SIZE')
	parser.add_argument('--WINDOW_SIZE')
	parser.add_argument('--RESULTS_ROOT')
	args = parser.parse_args()
	args = vars(args)

	for key, item in args.items():
		if item is not None:
			if key in ['LSTM_SAMPLE_SIZE', 'WINDOW_SIZE']:
				changed = list(map(int, item.strip('][').split(' ')))
				params[key] = changed
				print("changed params in ", key)
			elif key in ['RESULTS_ROOT']:
				params[key] = Path(item)
				print("changed params in ", key)
			else:
				changed = item.strip('][').split(' ')
				params[key] = changed
				print("changed params in ", key)

	return params

if __name__ == '__main__':
    #Initial parameters
    paramsDict = {'TESTED_USERS': ['jackie', 'Juan'],
              'ALL_USERS': ['juan', 'jackie'],
              'EEG_CHANNELS': ['FZ', 'F7', 'F3', 'F4', 'F8'],
              'POWER_COEFFICIENTS': ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
              'LSTM_SAMPLE_SIZE': [120, 135, 150],
              'WINDOW_SIZE': [10, 20, 30],
              'RESULTS_ROOT': Path('.').resolve() / 'results/EegResults/results_transfer9/exp04_30/' / 'OnlyJuanJackie'
              }
    # Read command line arguments if any
    paramsDict = readCommandLineArg(paramsDict)
    #Turn params into a simpleNamespace
    params = SimpleNamespace(**paramsDict)
    #Load main modules
    crossValidationModule = CrossValidationRoutines()
    utilsModule = Utils()

    #Main Script
    for lstmSampleSize,windowSize in itertools.product(params.LSTM_SAMPLE_SIZE, params.WINDOW_SIZE):

        lstmSteps = int(lstmSampleSize/windowSize)
        completeResults = []

        for user in params.TESTED_USERS:
            dataPath = Path('./data/DifferentWindowSizeData/{:02d}s/'.format(windowSize))
            resultsPath = params.RESULTS_ROOT /'window{:02d}s_sampleSize{:02d}s'.format(windowSize,lstmSampleSize)

            if not resultsPath.exists():
                print('create ', resultsPath)
                Path.mkdir(resultsPath,parents=True)

            results = crossValidationModule.userCrossValidationMultiUser(lstmSteps, dataPath,resultsPath,
                                                                         user,params.ALL_USERS,eegChannels=params.EEG_CHANNELS,
                                                                         powerCoefficients=params.POWER_COEFFICIENTS)
            results.to_csv(resultsPath / '{:}_results.csv'.format(user), index= False)
            completeResults.append(results)

        completeResults = pd.concat(completeResults)
        completeResults.to_csv(resultsPath.parent / 'window{:02d}s_sampleSize{:02d}s.csv'.format(windowSize,lstmSampleSize) , index = False)
        utilsModule.sendSMS("Finished window{:02d}s_sampleSize{:02d}s".format(windowSize,lstmSampleSize))

    with open(resultsPath.parent / 'experimentParams.txt','w') as f:
        for k,i in paramsDict.items():
            line = str(k)+':'+str(i)+'\n'
            f.write(line)
