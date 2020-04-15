from pathlib import Path
import sys
sys.path.append(str(Path('./').resolve().parent.parent))
import PowerClassification.Utils.NetworkTraining as ut
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import heartpy as hp
import copy

nanCounter = 0

def getFeatures(signal,sampleFreq):
    w2, m2 = hp.process(signal, sample_rate=sampleFreq,high_precision=True)
    return w2, m2

def detectNan (m):
    for k,i in m.items():
        if  np.isnan(i):
            return True
    return False

def replaceNan(m, mPrev):
    for k,val in m.items():
        if  np.isnan(val):
            m[k] = mPrev[k].item()
        #Remove numpy masked constants not exactly sure if I am solving correctly this issue
        #Read about masked constants
        elif str(val) == '--':
            m[k] = mPrev[k].item()
    return m

def calculateShimmerFeatures(windowArray,df, sampleFreq,fileName):
    counter = 0
    dataCont = []

    for i in range(windowArray.shape[0]):
        v1 = windowArray[i]
        data = df[int(v1[1]):int(v1[2] + 1)]
        ppgSignal = data['PPG'].values
        label = int(df['label'].values.mean())
        counter += 1
        if v1[3] > 20: #[0.7, 3.5]
            # filtered = hp.scale_data(ppgSignal)
            filtered = hp.filter_signal(ppgSignal, [0.7, 3.5], sample_rate=sampleFreq, order=3, filtertype='bandpass')

            # windowSize = 3.0 #30Seconds samples
            windowSize = 1.4 #60seconds samples
            try:
                wd, m = hp.process(filtered, sample_rate=sampleFreq ,
                                   windowsize=windowSize,
                                    high_precision=True)
            except Exception as e:
                print(e)
                wd, m = hp.process(filtered, sample_rate=sampleFreq,  windowsize=windowSize)


            m['label'] = 0.0 if label == 5 else 1.0
            m['counter'] = i
            dataCont.append(copy.deepcopy(m))

            global nanCounter
            if detectNan(m):
                nanCounter += 1
                # hp.plotter(wd, m)
                print(nanCounter)

    result = pd.DataFrame(columns=m.keys())
    result = result.append(dataCont,True)
    return result

def calculateShimmerFeaturesV2(data, sampleFreq, fileName):
    ppgSignal = data['PPG'].values
    label = int(data['label'].values.mean())

    filtered = hp.filter_signal(ppgSignal, [0.7, 3.5], sample_rate=sampleFreq, order=3, filtertype='bandpass')

    # windowSize = 3.0 #30Seconds samples
    windowSize = 1.4 #60seconds samples
    try:
        wd, m = hp.process(filtered, sample_rate=sampleFreq ,
                           windowsize=windowSize,
                            high_precision=True)
    except Exception as e:
        print(e)
        wd, m = hp.process(filtered, sample_rate=sampleFreq,  windowsize=windowSize)

    m['label'] = 0.0 if label == 5 else 1.0

    global nanCounter
    if detectNan(m):
        nanCounter += 1
        # hp.plotter(wd, m)
        print(nanCounter)

    return m #Metrics from window

def main():
    dataPath = Path('./../data/')
    rawDataPath = dataPath / 'shimmer_raw_data'
    dstPath = dataPath / 'ShimmerPreprocessed'

    windowSize = [60]
    overlap = 30

    # Create Directory where all the data is going to be stored
    utilities = ut.Utils()
    utilities.makeDir(dstPath)

    for w1 in windowSize:

        utilities.makeDir(dstPath / '{:02d}s'.format(w1))

        # Check all the raw files and create a file with the specified data file
        for f2 in rawDataPath.rglob(('*.txt')):
            dstPathFinal = dstPath / '{:02d}s/{:}'.format(w1, f2.parent.name)
            user = f2.parent.name
            finalDataContainer = []

            # if u1 == 'juan':
            #     continue
            if not Path.exists(dstPathFinal):
                utilities.makeDir(dstPathFinal)

            # Get trial and session. Open data with pandas
            trial = re.findall("S[0-9]_T[0-9]", f2.name)
            trial = int(trial[0][-1])
            session = re.findall("S[0-9]_T[0-9]", f2.name)
            session = int(session[0][-4])
            df = pd.read_csv(f2, sep=',')

            print(trial, session, f2.name)

            # Initial values
            result = df.loc[(df['markerValue'] == 'not_active') &
                            (df['5secondWindow'] == 'WindowStart')]
            startIdx = result.index.values[0]
            startTime = result['ComputerTime'].values[0]
            df['Time'] = df['ComputerTime'] - startTime

            windowCounter = 0
            windowArray = []
            remainingData = copy.deepcopy(df)
            while True:
                # Get window
                window = remainingData.loc[(remainingData['ComputerTime'] > startTime) \
                                            & (remainingData['ComputerTime'] < (startTime + w1))]

                if window['ComputerTime'].values[-1]- window['ComputerTime'].values[0] <= w1*0.9: # Check if end was reached
                    print('remainder',window.shape[0])
                    break
                else:
                    beginTime = window['ComputerTime'].values[0]
                    endTime = window['ComputerTime'].values[-1]
                    generalInfo = {'User': user, 'Session': session, 'Trial': trial,
                                   'BeginTime':beginTime,'EndTime':endTime, 'length': (endTime-beginTime),
                                   'WindowNumber':windowCounter}
                    features = calculateShimmerFeaturesV2(window,sampleFreq=204.8,fileName=f2.name)

                    #If a nan is detected replaced the nan values with the values in the previous window
                    if detectNan(features):
                        #This will fail if the first window is the one who contains the nan values.
                        #You need to think in a better solution.
                        features = replaceNan(features, finalDataContainer[-1])

                    dataDict = dict(generalInfo,**features)
                    finalDataContainer.append(pd.DataFrame(data=[list(dataDict.values())],columns=dataDict.keys()))

                # Update values
                windowCounter += 1
                startTime = (startTime + w1 -overlap)
                remainingData = remainingData.loc[remainingData['ComputerTime'] > startTime]


            finalDataContainer = pd.concat(finalDataContainer, ignore_index=True)
            pf = dstPathFinal / '{:}_S{:d}_T{:}_Shimmer.txt'.format(user,session,trial)
            finalDataContainer.to_csv(pf, sep=',', index=False)

            print(pf)

        print("Number of nan", nanCounter)
        print("Nan elements where replaced with values of the previous window")
        print("This is the best solution I have found so far; however, can fail easily if the first window has the Nan values")

if __name__ == '__main__':
    main()

