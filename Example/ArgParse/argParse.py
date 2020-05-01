"""
Argparse example

"""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import argparse
import mne
import tensorflow as tf
import time

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

	for k, i in args.items():
		if i is not None:
			if k in ['LSTM_SAMPLE_SIZE','WINDOW_SIZE']:
				changed = list( map(int, i.strip('][').split(' ')) )
				params[k] = changed
				print("changed params in ", k)
			elif k in ['RESULTS_ROOT']:
				params[k] = Path(i)
				print("changed params in ", k)
			else:
				changed = i.strip('][').split(' ')
				params[k] = changed
				print("changed params in ", k)

	return params


if __name__ == '__main__':
	params = {  'TESTED_USERS': ['jackie'],
				'ALL_USERS' : ['ryan','juan', 'jackie','jhony'],
				'EEG_CHANNELS' : ['FZ', 'F7', 'F3', 'F4', 'F8'],
				'POWER_COEFFICIENTS':  ['Low', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
				'LSTM_SAMPLE_SIZE' : [120],
				'WINDOW_SIZE' : [10],
				'RESULTS_ROOT' : Path('./results/EegResults/results_transfer9/') / 'temp/temp2'
		}

	#Read command line arguments if any
	params = readCommandLineArg(params)


	time.sleep(4)
	params = SimpleNamespace(**params)
	print(params.TESTED_USERS)
	print(params.ALL_USERS)
	time.sleep(4)
	print(params.LSTM_SAMPLE_SIZE)
	print(params.RESULTS_ROOT)