from pathlib import Path
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt

def getPpgFeatures(ppg, fs = 204.8,):
    filtered = hp.filter_signal(data, [0.7, 3.5], sample_rate=fs, order=3, filtertype='bandpass')
    wd, m = hp.process(filtered, sample_rate=fs, calc_freq=True)

    # # display measures computed
    # for measure in m.keys():
    #     print('%s: %f' % (measure, m[measure]))
    #
    # # call plotter
    # hp.plotter(wd, m)

    return m

if __name__ == '__main__':
    fs = 204.8
    user = 'jhony'
    p = Path('.').resolve()
    featuresFrame = None


    dataPath = p.parent / 'data/shimmer_raw_data'  / user

    for file in dataPath.glob('*.txt'):
        print(file.name)
        df = pd.read_csv(file,sep=',')
        data, timer, label = df['PPG'].values[:-1], df['ComputerTime'].values, int(df['label'].values.mean())
        m = getPpgFeatures(data,fs=fs)

        m['label'] = 'low' if label == 5 else 'high'
        m['user'] = user
        if featuresFrame is None:
            columns = list(m.keys())
            featuresFrame = pd.DataFrame(columns=columns)

        featuresFrame = featuresFrame.append(m,True)
        q =0
    print('t')
    featuresFrame.to_csv(p/(user+'features.csv'),index=False)
