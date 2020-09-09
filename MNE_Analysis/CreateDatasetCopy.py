import re
import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName

# Check all the raw files and create a file with the specified data file
srcPath =  Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiment1-Pilot').resolve()
dstBasePath =  Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiment1-Pilot-Final').resolve()
lowTaskEpochs  = []
highTaskEpochs = []

#Sessions black list
black_list = {'UI01':['1','6','3','7'],
              'UI02':['7','4','2'],
              'UI03':['2'],
              'UI04':['4'],
              'UI05':['3'],
              'UI06':['2'],
              'UI07':[''],
              'UI08':[''],}

if __name__ == '__main__':

    for file in srcPath.rglob(('*.edf')):

        # Rename files --> remove identifiers
        uid = re.findall('.+(?=_S[0-9]_T[0-9]_)', file.name)[0]
        session = re.findall('(?<=_S)[0-9](?=_T[0-9]_)', file.name)[0]
        trial = re.findall('(?<=_S[0-9]_T)[0-9](?=_)', file.name)[0]
        task = re.findall('(?<=_S[0-9]_T[0-9]_).+(?=_)', file.name)[0]
        preprocess = re.findall('(?<=_{:}_).+(?=\.edf)'.format(task), file.name)[0]

        if session in black_list or preprocess == 'pyprep':
            continue

        dstFile = dstBasePath / uid / preprocess
        if not dstFile.exists():
            dstFile.mkdir(parents=True)

        dstFile = dstFile / file.name

        print(uid, session, trial, task, preprocess)

        shutil.copy(str(file), str(dstFile))