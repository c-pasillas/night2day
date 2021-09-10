#attempt to run model

#import statements
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import math
import time
import tensorflow.keras.layers as layers
import pandas as pd
import IPython.display as dis
from PIL import Image
import datetime as date
import FNN_train

def read_channels(path):
    with open (path) as f:
        lines = f.readlines()
        PREDICTAND = lines[0].strip()
        PREDICTORS = lines[1].strip().split()
        return PREDICTORS, PREDICTAND

def function(model, case, PREDICTORS, PREDICTAND):
    
    tors = np.stack([case[c] for c in PREDICTORS], axis = -1) #X in array
    tand = case[PREDICTAND] #y in array, 

    TORS = tors.reshape(-1, len(PREDICTORS))
    TAND = tand.flatten()
    
    o_shape = tand.shape

    #lets predict on all the data
    print("stacked channels now predicting")
    model_output= model.predict(TORS)
    model_outputfinal= model_output.reshape(o_shape)
    return model_outputfinal

def predict(args):
    print("i am in predict and args are ", args)
    case = np.load(args.npz_path)
    print("I loaded the case")
    model = tf.keras.models.load_model(args.model_path)#
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

