#makes just the predicotrs/tand so can do train seperately
#FNNtrain
#import statements
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import math
import tensorflow.keras.layers as layers
import pandas as pd
import IPython.display as dis
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import aoi
from common import log, rgb, reset, blue, orange, yellow, bold

def write_channels(path, inputfile, PREDICTORS, PREDICTAND):
    with open (path, "w") as f:
        print(inputfile, file =f)
        print(PREDICTAND[0], file = f)
        for p in PREDICTORS:
            print(p, file = f, end = " ")
        #for p in TORS:
         #   print(p, file = f, end = "\n")   
        print(file =f)

def set_up(case, PREDICTORS):
    tors = np.stack([case[c] for c in PREDICTORS], axis = -1) #X in array
    tand = case['SM_reflectance'] #y in array, 
    print("old min/max are",tand.min(), tand.max())
    tand =  (np.clip(tand, 0, 100)) / 100 #adjuates for any over 100% reflectance (ie  city lights) and also makes 0-1 vs 0-100
    print("new min/max are",tand.min(), tand.max())
    TORS = tors.reshape(-1, len(PREDICTORS))
    TAND = tand.flatten()
    print(TORS.shape)
    print(TAND.shape)
    #split the data to test train
    
    # do the reflectance reduction math here ( ie take 1/3 of ref valeus LT 10% and 1/3 = GT 100% retain 100% between 10 and 99
    #training split
    ts = 0.2
    #random seed
    rs = 10
    
    return(TORS, TAND,ts, rs)
   


def FNN_prep(args):
    print("im in FNN prep and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    log.info(f'I am making the inputs')
    all_predictors = ['M13norm', 'M14norm', 'M15norm', 'M16norm', 'C07norm', 'C11norm', 'C13norm', 'C15norm', 'normBTD_M13M14', 'normBTD_M13M15', 'normBTD_M13M16', 'normBTD_M14M15', 'normBTD_M14M16', 'normBTD_M15M16', 'normBTD_C07C11', 'normBTD_C07C13', 'normBTD_C07C15', 'normBTD_C11C13', 'normBTD_C11C15', 'normBTD_C13C15']
    
    if args.Predictors:
        PREDICTORS = args.Predictors
        n_input = len(args.Predictors)
    else:
        PREDICTORS =[b for b in all_predictors if b in case]
        n_input = len(PREDICTORS)
        print("n_inputs is ", n_input, PREDICTORS) #hardcoded need to fix
    PREDICTAND = ['SM_reflectance']
    print("my predictand is", PREDICTAND)
    print("my predictors are", PREDICTORS)
    TORS, TAND, ts, rs = set_up(case, PREDICTORS)
    print("i am starting train_test_split")
    TORS_train, TORS_test, TAND_train, TAND_test  = train_test_split(TORS, TAND, test_size=ts, random_state=rs)
    
    #save the channels
    log.info(f' i am saving the channels')
    write_channels(f"FNN_tors_tand_{made}", args.npz_path, PREDICTORS, PREDICTAND)
    log.info(f' i am saving the tors/tands for retrain')
    savepath = args.npz_path[:-4]+ f"TORS_TAND_ONLY{made}.npz"
    savepath2=  args.npz_path[:-4]+ f"TORS_TAND_train_test{made}.npz"
    np.savez_compressed(savepath, TORS, TAND)
    np.savez_compressed(savepath2, TORS_train, TORS_test, TAND_train, TAND_test)
    log.info(f'done with making tors/tand')
  
    
