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
from common import log, rgb, reset, blue, orange, yellow, bold



        
def write_channels_AHI(path, inputfile, PREDICTORS):
    with open (path, "w") as f:
        print(inputfile, file =f)
        for p in PREDICTORS:
            print(p, file = f, end = " ")
        #for p in TORS:
         #   print(p, file = f, end = "\n")   
        print(file =f)


def read_channels(path):
    with open (path) as f:
        lines = f.readlines()
        PREDICTORS = lines[2].strip().split()
        return PREDICTORS

def function_AHI(model, case, PREDICTORS):
    # for Cbands normal 7,11,13,15
    #Cbands = ["C07norm", "C11norm", "C13norm", "C15norm", "normBTD_C07C11", "normBTD_C07C13", "normBTD_C07C15", "normBTD_C11C13",  "normBTD_C11C15", "normBTD_C13C15" ] 
    #using C14 instead of C13
    #Cbands = ["C07norm", "C11norm", "C14norm", "C15norm", "normBTD_C07C11", "normBTD_C07C14", "normBTD_C07C15", "normBTD_C11C14", "normBTD_C11C15","normBTD_C14C15", ] 
    #using all
    Bbands = ["B07norm", "B11norm", "B13norm","B14norm", "B15norm", "normBTD_B07B11","normBTD_B07B13", "normBTD_B07B14", "normBTD_B07B15", "normBTD_B11B13","normBTD_B11B14", "normBTD_B11B15",'normBTD_B13B14',"normBTD_B13B15", "normBTD_B14B15"] 
    
     
    tandish =case['B07norm']#just to get the orignal shape back
    tors = np.stack([case[c] for c in Bbands], axis = -1) #X in array

    TORS = tors.reshape(-1, len(PREDICTORS))
    TORS9999 = np.nan_to_num(TORS, copy=True, nan=9999, posinf=None, neginf=None)
    
    o_shape = tandish.shape

        #lets predict on all the data
    print("stacked channels now predicting on")
    log.info(f' model.predict has started')
    model_output= model.predict(TORS9999)
        #replace the 9999 with NANs 
    log.info(f' model.predict has ended')
    print("im done predicting")
    NANindex =np.isnan(TORS).any(axis =1) 
    model_output[NANindex] = np.nan
    model_outputfinal= model_output.reshape(o_shape)
    return model_outputfinal
    
    

    

def predict(args):
    print("i am in predict and args are ", args)
    case = np.load(args.npz_path)
    print("I loaded the case")
    print("the files in the case are", case.files)
    print("args.modelpath is", args.model_path)
    model = tf.keras.models.load_model(args.model_path)#
    print("I loaded the model")
    #only need predictors now
    PREDICTORS = read_channels(args.channel_path)
    print("my predictors are", PREDICTORS)
    print("I loaded the channels list")
   #if to split the ABI and GOES is here 
    #if args.pixel:
    MLvalues = function_AHI(model, case, PREDICTORS)
    print("I am now saving ML values")
    #save the ML truths 
    made =time.strftime("%Y-%m-%dT%H%M")     
    #TODO make this a folder path resovled
    #if different folders need to remvoe the fodler path
    #savepath = f"ML_truth_C13BAND_{PREDICTORS[2]}_{args.npz_path[-27:-4]}_MODELis_{args.model_path}.npz"
    #savepath = f"ML_truth_C13BAND_{PREDICTORS[2]}_CSUsampleGOES_MODELis_{args.model_path}.npz"
    savepath = f"ML_truth_DATA_{args.npz_path[:-4]}_MODEL_{args.model_path}_{made}.npz" #[0:9]  
    #savepath = f"ML_truth_{made}_BY_{args.npz_path[:-4]}_FROM_FNN_C2019_2021-12-06T1801.npz" #[0:9]   
    np.savez_compressed(savepath, MLtruth = MLvalues, channel = [PREDICTORS])

    #log.info(f' i am saving the channels')
    #write_channels(f"FNN_predict_channels_{args.npz_path[:-4]}_{made}", args.npz_path, PREDICTORS, PREDICTAND)
