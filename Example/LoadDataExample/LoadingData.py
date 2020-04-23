import copy
from itertools import product
from pathlib import Path
import pandas as pd
import re

EEG_channels = ["F7","F3","FZ","F4","F8"]

Power_coefficients = ['Theta','Alpha','Beta','Gamma']

newColumnNames = [x+'-'+y for x,y in product(EEG_channels, Power_coefficients)]

if __name__ == "__main__":

    dataDir = Path('./../../').resolve() / 'PowerClassification' / 'data'/ 'DifferentWindowSizeData'/ '60s'
    dataDir = dataDir
    users = ['juan','juanBaseline']
    container = []
    for u in users:
        p = dataDir / u
        print(p)
        for f in p.rglob('*.txt'):
            session = int(re.findall('(?<=_S)[0-9](?=_T)', f.name)[0])
            trial =   int(re.findall('(?<=_T)[0-9](?=_p)', f.name)[0])

            if session in [4,5,6,7]:
                dataFrame = pd.read_csv(f,index_col=0)
                dataFrame['Session'] = session
                dataFrame['Trial'] = trial

                dataFrame = dataFrame[['Session','Trial', 'Label']+newColumnNames]
                container.append(copy.deepcopy(dataFrame))

    x=0
    container = pd.concat(container)
    container = container.sort_values(["Session", "Trial"], ascending = (True, True))
    container.to_csv('ie533_project.csv', index=False)

    print(dataDir)
