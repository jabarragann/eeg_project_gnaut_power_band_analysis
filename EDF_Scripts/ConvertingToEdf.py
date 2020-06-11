import pandas as pd
from pyedflib import highlevel
import pyedflib
import numpy as np
import os
import re

EEG_channels = [ "FP1","FP2","AF3","AF4","F7","F3","FZ","F4",
                  "F8","FC5","FC1","FC2","FC6","T7","C3","CZ",
                  "C4","T8","CP5","CP1","CP2","CP6","P7","P3",
                  "PZ","P4","P8","PO3","PO4","OZ"]

if __name__ == '__main__':
    path = './raw/'
    files = os.listdir(path)

    for f in files:
        user = re.findall('^[A-Za-z]+(?=_)', f)[0]
        session = re.findall('[0-9]+(?=_T)',f)[0]
        trial = re.findall('[0-9]+(?=_epoc)', f)[0]

        destinationFile = './converted/{:}_S{:}_T{:}.edf'.format(user,session,trial)
        sourceFile = path+f

        #raw data information
        dataframe = pd.read_csv(sourceFile.format(user),delimiter=',')
        signals = dataframe[EEG_channels].values.transpose()

        #Event processing
        events = dataframe[['COMPUTER_TIME','markerValue']]
        initTime = events['COMPUTER_TIME'].values[0]

        startAndFinish = events[np.logical_or((dataframe['markerValue'] == 'started').values, (dataframe['markerValue'] == 'finished').values)]
        temp = startAndFinish['COMPUTER_TIME'].apply(lambda x:x-initTime)
        startAndFinish.insert(0, "time", temp, True)

        #Create writer
        writer = pyedflib.EdfWriter(destinationFile, len(EEG_channels), file_type=1)

        #Create header
        writer.setPatientName(user)

        #Signals
        signal_headers = highlevel.make_signal_headers(EEG_channels, sample_rate=250)
        writer.setSignalHeaders(signal_headers)
        writer.writeSamples(signals)

        #Write Starting and ending events
        for index, row in startAndFinish.iterrows():
            writer.writeAnnotation(row['time'], 5, row['markerValue'], str_format='utf-8')

        writer.close()



