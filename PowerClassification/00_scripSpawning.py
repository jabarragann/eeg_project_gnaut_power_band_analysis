import subprocess
from pathlib import Path

def execute(cmd):
	process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True)

	output = process.stdout
	return output

if __name__ == '__main__':

	resultsPath = Path('.').resolve() / 'results/EegResults/results_transfer9/exp04_30/'
	print(resultsPath)

	if not resultsPath.exists():
		resultsPath.mkdir(parents=True)


	cmd1 = [  'python','./SimpleLstmTransferTest9.py',
			  '--TESTED_USERS' , '[Juan Jackie]',
			  '--ALL_USERS'    , '[Juan Jackie]',
			  '--RESULTS_ROOT', str(resultsPath / 'OnlyJuanJackie'),
			  '--LSTM_SAMPLE_SIZE','[120 135 150]',
			  '--WINDOW_SIZE', '[10 20 30]',
			  ]

	cmd2 = ['python', './SimpleLstmTransferTest9.py',
			'--TESTED_USERS', '[Juan Jackie Jhony]',
			'--ALL_USERS', '[Juan Jackie Jhony]',
			'--RESULTS_ROOT', str(resultsPath / 'OnlyJuanJackieJhony'),
			'--LSTM_SAMPLE_SIZE', '[120 135 150]',
			'--WINDOW_SIZE', '[10 20 30]',
			]

	cmd3 = ['python', './SimpleLstmTransferTest9.py',
			'--TESTED_USERS', '[Juan Jackie Jhony Ryan]',
			'--ALL_USERS', '[Juan Jackie Jhony Ryan]',
			'--RESULTS_ROOT', str(resultsPath / 'PowerBands_LDTABG'),
			'--LSTM_SAMPLE_SIZE', '[120 135 150]',
			'--WINDOW_SIZE', '[10 20 30]',
			'--POWER_COEFFICIENTS', '[Low Delta Theta Alpha Beta Gamma]'
			]

	cmd4 = ['python', './SimpleLstmTransferTest9.py',
			'--TESTED_USERS', '[Juan Jackie Jhony Ryan]',
			'--ALL_USERS', '[Juan Jackie Jhony Ryan]',
			'--RESULTS_ROOT', str(resultsPath / 'PowerBands_DTABG'),
			'--LSTM_SAMPLE_SIZE', '[120 135 150]',
			'--WINDOW_SIZE', '[10 20 30]',
			'--POWER_COEFFICIENTS', '[Delta Theta Alpha Beta Gamma]'
			]

	cmd5 = ['python', './SimpleLstmTransferTest9.py',
			'--TESTED_USERS', '[Juan Jackie Jhony Ryan]',
			'--ALL_USERS', '[Juan Jackie Jhony Ryan]',
			'--RESULTS_ROOT', str(resultsPath / 'PowerBands_TABG'),
			'--LSTM_SAMPLE_SIZE', '[120 135 150]',
			'--WINDOW_SIZE', '[10 20 30]',
			'--POWER_COEFFICIENTS', '[Theta Alpha Beta Gamma]'
			]

	commands = [cmd1, cmd2, cmd3, cmd4, cmd5]

	for c in commands:
		print("Executing")
		print(c)
		out = execute(cmd2)