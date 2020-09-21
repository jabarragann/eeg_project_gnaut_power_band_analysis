from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


userNameMapping  = {'jackie':'Subject 1','ryan':'Subject 2', 'juan':'Subject 3',
                    'jhony' :'Subject 4','karuna':'Subject 5','santy':'Subject 6'}

def formatDataForAnalysis(path):
    df = pd.read_csv(path)

    # Format data - TestAccBefore as proportionOfTransfer =0
    users = df.User.unique()
    compilation = []
    for u in users:
        partOfData = df.loc[df['User'] == u].reset_index(drop=True)
        testSessions = partOfData.TestSession.unique()
        for sess in testSessions:
            tinyPart = partOfData.loc[partOfData['TestSession'] == sess].reset_index(drop=True)
            testAccBefore = tinyPart.loc[0, 'TestAccBefore']
            newRow = {'User': u, 'TestSession': sess, 'proportionOfTransfer': 0.0,
                      'TestAccAfter': testAccBefore}
            tinyPart = tinyPart.append(newRow, ignore_index=True)
            compilation.append(tinyPart)

    return pd.concat(compilation)

if __name__ == '__main__':

    # Load results data
    timePath = Path('C:\\Users\\asus\\PycharmProjects\\eeg_project_gnaut_power_band_analysis\\TimeClassification\\results\\results_transfer10\\aa11_pyprep\\window10s_sampleSize140s.csv')
    freqPath = Path(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\results\EegResults\results_transfer10\aa13a_pyprep_complete\window10s_sampleSize110s.csv')
    assert timePath.exists(), "Make sure time path is correctly set"
    assert freqPath.exists(), "Make sure freq path are correctly set"
    timeData = formatDataForAnalysis(timePath)
    timeData['ModelType'] = "TimeModel"
    freqData = formatDataForAnalysis(freqPath)
    freqData['ModelType'] = "FreqModel"

    combined = [freqData, timeData]
    combined = pd.concat(combined)
    combined.to_csv("./Data/combinedForStat.csv")

    #Get Variance data
    varCombined = []
    for df in [freqData, timeData]:
        users = df.User.unique()
        for u in users:
            p = df.loc[df.User == u]
            modelType = p.ModelType.values[0]
            p = p.groupby(by="proportionOfTransfer").std()
            p['User'] = u
            p['ModelType'] = modelType
            varCombined.append(p)

    varCombined = pd.concat(varCombined).reset_index()
    varCombined.rename({'TestAccAfter': 'TestAccAfterVar'},axis='columns', inplace=True)
    varCombined.to_csv("./Data/combinedForStatVar.csv")



# timeData = pd.read_csv(timePath)
    # timeData['ModelType'] = "TimeModel"
    # freqData = pd.read_csv(freqPath)
    # freqData['ModelType'] = "FreqModel"

    # #Format data - TestAccBefore as proportionOfTransfer =0
    # users = freqData.User.unique()
    # compile = []
    # for u in users:
    #     partOfData = freqData.loc[freqData['User'] == u].reset_index(drop=True)
    #     testSessions = partOfData.TestSession.unique()
    #     for sess in testSessions:
    #         tinyPart = partOfData.loc[partOfData['TestSession'] == sess].reset_index(drop=True)
    #         testAccBefore = tinyPart.loc[0,'TestAccBefore']
    #         newRow = {'User': u,'TestSession':sess,'proportionOfTransfer': 0.0,
    #                   'TestAccAfter':testAccBefore}
    #         tinyPart = tinyPart.append(newRow, ignore_index=True)
    #         compile.append(tinyPart)