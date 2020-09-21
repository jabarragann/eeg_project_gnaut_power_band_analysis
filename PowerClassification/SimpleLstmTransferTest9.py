'''
Grid search for the best lstm sample size and windows size.

multi user
'''

from pathlib import Path
import sys
sys.path.append(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis')
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

#['FZ', 'F7', 'F3', 'F4', 'F8'],
#['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
# 'EEG_CHANNELS': ["FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
#                  "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
#                  "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
#                  "PZ", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "OZ"]
#'UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08'
#[11,25,50,75,100,125,150,175]
#['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']
if __name__ == '__main__':
    #Initial parameters
    paramsDict = {'TESTED_USERS': ['UI05'],
              'ALL_USERS': ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08'],
              'EEG_CHANNELS': [   "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                                  "PZ","P4","P8","PO3","PO4","OZ"], #PO8 and PO7 removed
            #   'EEG_CHANNELS':['AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8',
            # 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'CP5', 'CP6'],
              'POWER_COEFFICIENTS': ['Delta', 'Theta', 'Alpha', 'Beta'],
              'LSTM_SAMPLE_SIZE': [150],
              'WINDOW_SIZE': [5],
              'RESULTS_ROOT': Path('.').resolve() / 'results/EegResults/results_transfer9/aa16b_pyprep_complete/',
              'SOURCE_PATH': './data/de-identified-pyprep-dataset-reduced-critically-exp/',
              'AIM':"150 - 5 - fullset of channels"
              }
    # Read command line arguments if any
    paramsDict = readCommandLineArg(paramsDict)
    #Turn params into a simpleNamespace
    params = SimpleNamespace(**paramsDict)
    #Load main modules
    crossValidationModule = CrossValidationRoutines()
    utilsModule = Utils()

    p = params.RESULTS_ROOT
    #Print parameter dict
    for k, i in paramsDict.items():
        print(k, i)
    #Main Script
    for lstmSampleSize,windowSize in itertools.product(params.LSTM_SAMPLE_SIZE, params.WINDOW_SIZE):

        if windowSize > lstmSampleSize:
            print("skip window size {:} and sample size {:}".format(windowSize, lstmSampleSize))
            continue

        lstmSteps = int(lstmSampleSize/windowSize)
        completeResults = []

        for user in params.TESTED_USERS:
            dataPath = Path(params.SOURCE_PATH)/'{:02d}s/'.format(windowSize)
            resultsPath = params.RESULTS_ROOT /'window{:02d}s_sampleSize{:02d}s'.format(windowSize,lstmSampleSize)

            if not resultsPath.exists():
                print('create ', resultsPath)
                Path.mkdir(resultsPath,parents=True)

            with open(resultsPath/'readme.md','w') as f:
                f.write(params.AIM + '\n')

            results = crossValidationModule.userCrossValidationMultiUser(lstmSteps, dataPath,resultsPath,
                                                                         user,params.ALL_USERS,eegChannels=params.EEG_CHANNELS,
                                                                         powerCoefficients=params.POWER_COEFFICIENTS)
            results.to_csv(resultsPath / '{:}_results.csv'.format(user), index= False)
            completeResults.append(results)

        completeResults = pd.concat(completeResults)
        completeResults.to_csv(resultsPath.parent / 'window{:02d}s_sampleSize{:02d}s.csv'.format(windowSize,lstmSampleSize) , index = False)
        # utilsModule.sendSMS("Finished window{:02d}s_sampleSize{:02d}s".format(windowSize,lstmSampleSize))

    with open(resultsPath.parent / 'experimentParams.txt','w') as f:
        for k,i in paramsDict.items():
            line = str(k)+':'+str(i)+'\n'
            f.write(line)
