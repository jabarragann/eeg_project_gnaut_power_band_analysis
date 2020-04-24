from PowerClassification.Utils import ShimmerModule as sm
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    user = 'juan'
    # listOfUsers = ['juan', 'jackie','ryan', 'jhony']
    listOfUsers = ['jackie']
    crossValidation = sm.CrossValidationRoutines()
    factoryModule = sm.NetworkFactoryModule()

    #Create a model
    model = factoryModule.createModel(6)
    model.summary()

    allResults = []
    resultsDir = Path('.').resolve() / 'results/simpleAnn9-ManualFeat'

    for user in listOfUsers:
        dataDir =  Path('./../').resolve() / "data" / "ShimmerPreprocessed" / "manual" /"60s"
        plotsDir =  resultsDir / user

        if not plotsDir.exists():
            plotsDir.mkdir(parents=True)

        print(dataDir)
        print(dataDir.exists())
        df = crossValidation.userCrossValidationMultiUser(dataDir, plotsDir, user, listOfUsers)
        df.to_csv(plotsDir/'results.csv', index= False)
        allResults.append(df)

    fdf =  pd.concat(allResults)

    fdf.to_csv(resultsDir / 'results.csv', index=False)
