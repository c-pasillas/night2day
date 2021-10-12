#!/usr/bin/env python
# coding: utf-8


#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import gzip
import math
import tensorflow.keras.layers as layers
import pandas as pd
import IPython.display as dis
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import aoi
import pprint
import matplotlib as mpl
import importlib
from sklearn import linear_model
from sklearn import metrics
import os
import sys
import seaborn as sb
from matplotlib import style
import pandas as pd
import itertools as it
from sklearn.preprocessing import StandardScaler
import pickle
scaler = StandardScaler()
from common import log, bold, reset, yellow, blue, orange, rgb
from pathlib import Path
from datetime import datetime

####helper functions

def write_channels(path, inputfile, TAND, TORS):
    with open (path, "w") as f:
        print(inputfile, file =f)
        print(TAND, file = f)
        for p in TORS:
            print(p, file = f, end = " ")
        print(file =f)

def set_up(case, predictors, predictand):
    tors = np.stack([case[c] for c in predictors], axis = -1) #X in array
    tand = case[predictand] #y in array, 

    TORS = tors.reshape(-1, len(predictors))
    TAND = tand.flatten()

    #split the data to test train
    #training split
    ts = 0.2
    #random seed
    rs = 10
    
    return train_test_split(TORS, TAND, test_size=ts, random_state=rs)


###main

def MLR_train(args):
    print("im in MLR train and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    log.info(f'I am making the inputs')
    TORS_train, TORS_test, TAND_train, TAND_test  = set_up(case, args.Predictors, args.DNB)
    log.info(f'I am now making the model')
    n_input = len(args.Predictors)
    
    X = TORS_train
    Y = TAND_train
    
    log.info("fitting MLR")
    regOLS = linear_model.LinearRegression(normalize=False, fit_intercept=True)   
    regOLS.fit(X, Y)

    #MLR_truths = regOLS.predict(X) --shoudlnt be here becuase then doing ML based on the same inputs that trained the model

    log.info(f'I am now saving the model')
    made =time.strftime("%Y-%m-%dT%H%M")
    model.save(f'MLR_{args.npz_path[:11]}_{made}')

    #save the history as a text file
    with open (f"myhistory_{made}", "w") as f:
        import pprint
        pprint.pprint(history.history, stream =f)
    #save the channels
    write_channels(f"MLR_model_channels_{made}", args.npz_path, args.DNB, args.Predictors)
    log.info(f'done with MLR model training')