import pandas as pd
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

userList  = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']

if __name__ =='__main__':

    resultsDir = 'aa13a_deidentified_pyprep_complete'
    path = Path('./').resolve().parent / 'results' / 'EegResults' /'results_transfer9' / resultsDir
    dataSummary = []
    for file in path.glob('*.csv'):
        windowSize = int(re.findall('(?<=dow)[0-9]+(?=s)', file.name)[0][-2:])
        sampleSize = int(re.findall('(?<=Size)[0-9]+(?=s\.csv)', file.name)[0])

        #Load data
        df = pd.read_csv(file, sep = ',')
        df['WindowSize'] = windowSize
        df['ObservationSize'] = sampleSize
        dataSummary.append(df.copy())

    dataCombined = pd.concat(dataSummary)
    dataCombined.to_csv('-' +resultsDir+'.csv',sep=',')
