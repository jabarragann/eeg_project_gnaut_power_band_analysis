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
            u1 = f2.parent.name

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

            windowCounter = 0
            windowArray = []
            while True:
                result = df.loc[df['ComputerTime'] > (startTime + w1)]  # Get end Idx
                if result.shape[0] <= 1: # Check if end was reached
                    endIdx = df.index.values[-1]
                    endTime = df['ComputerTime'].values[-1]
                    t = [windowCounter, startIdx, endIdx, endTime - startTime]
                    windowArray.append(t)
                    break

                endIdx = result.index.values[0]
                endTime = result['ComputerTime'].values[0]
                t = [windowCounter, startIdx, endIdx, endTime - startTime]
                windowArray.append(t)

                # Update values
                windowCounter += 1
                startIdx = endIdx + 1
                result = df.loc[df['ComputerTime'] > (startTime + w1 - overlap)]
                startTime = result['ComputerTime'].values[0]

            windowArray = np.array(windowArray)

            shimmerFeatures = calculateShimmerFeatures(windowArray,df,204.8, fileName=f2.name)

            pf = dstPathFinal / '{:}_S{:d}_T{:}_time.txt'.format(u1,session,trial)
            shimmerFeatures.to_csv(pf, sep=',', index=False)

            print(pf)

        utilities.sendSMS('The dataset creation finished')
        print("Number of nan", nanCounter)

if __name__ == '__main__':
    main()

