import yasa
import pandas as pd
import numpy as  np

if __name__ == '__main__':

    #Sampling frequency
    sf = 100
    #Open txt file and get data
    with open('./data.txt','r') as f1:
        data = f1.readlines()
        data = list(map(lambda x:float(x),data))
        data = np.array([data,data])

    # Relative bandpower per channel on the whole recording (entire data)
    bd = yasa.bandpower(data, sf=sf, ch_names=['F3','F4'], win_sec=4,
                        bands=[(0.0, 0.5, 'Low'),(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                               (12, 30, 'Beta'), (30, 50, 'Gamma')])

    print(bd)
