#!/usr/bin/env python
# coding: utf-8


#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import gzip
import math
import time
import tensorflow.keras.layers as layers
import pandas as pd
import IPython.display as dis
from PIL import Image
import datetime as date
import FNN_train
from sklearn.model_selection import train_test_split
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


def read_channels(path):
    with open (path) as f:
        lines = f.readlines()
        PREDICTAND = lines[1].strip()
        PREDICTORS = lines[2].strip().split()
        return PREDICTORS, PREDICTAND

def function(model, case, PREDICTORS, PREDICTAND):
    
    tors = np.stack([case[c] for c in PREDICTORS], axis = -1) #X in array
    tand = case[PREDICTAND] #y in array, 

    TORS = tors.reshape(-1, len(PREDICTORS))
    TORS9999 = np.nan_to_num(TORS, copy=True, nan=9999, posinf=None, neginf=None)
    
    o_shape = tand.shape

    #lets predict on all the data
    print("stacked channels now predicting")
    model_output= model.predict(TORS9999)
    #replace the 9999 with NANs 
    print("im done predicting")
    NANindex =np.isnan(TORS).any(axis =1) 
    model_output[NANindex] = np.nan
    model_outputfinal= model_output.reshape(o_shape)
    return model_outputfinal


def predict(args):
    print("i am in predict and args are ", args)
    case = np.load(args.npz_path)
    print("I loaded the case")
    print("args.modelpath is", args.model_path)
    model = tf.keras.models.load_model(args.model_path)#
    #or is it this?
    
    #with open (model_path, 'rb') as g:
    #        model = pickle.load(g)
    print("I loaded the model")
      
    PREDICTORS, PREDICTAND = read_channels(args.channel_path)
    print("I loaded the channels list")
    MLvalues = function(model, case, PREDICTORS, PREDICTAND)
    print("I am now saving ML values")
    #save the ML truths 
    made =time.strftime("%Y-%m-%dT%H%M")     
    #TODO make this a folder path resovled
    savepath = args.npz_path[:9]+ f"_MLtruth_{made}.npz"
    np.savez(savepath, MLtruth = MLvalues, channel = [PREDICTAND])

