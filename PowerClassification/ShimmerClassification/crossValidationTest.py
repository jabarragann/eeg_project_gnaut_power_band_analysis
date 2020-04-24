import PowerClassification.Utils.ShimmerModule as shimmerMod
from pathlib import Path


if __name__ == '__main__':
    users = ['jackie','ryan','juan','jhony']
    resultsPath = Path('.')/'results'
    DATA_DIR = Path('./').resolve().parent / 'data' / 'shimmerPreprocessed' / '60s'
    crossVal = shimmerMod.CrossValidationRoutines()

    results = crossVal.userCrossValidationMultiUser(DATA_DIR,resultsPath,'juan',users)
    results.to_csv(resultsPath/'results.csv',index=False)