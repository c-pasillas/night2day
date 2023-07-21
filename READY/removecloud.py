#attempt to run model
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
    
    #for testing make these much smaller
    TORS = TORS[:100]
    TAND = TAND[:100]
    print(f'reduced lengths is {(TAND.shape)} and {(TORS.shape)}')
    # do the reflectance reduction math here ( ie take 1/3 of ref valeus LT 10% and 1/3 = GT 100% retain 100% between 10 and 99
    #training split
    

    #TAND_idx = np.where(TAND < 0.2)
    #np.random.shuffle(TAND_idx)
    #print(len(TAND_idx))
    #print(int(len(TAND_idx)*.3))
    #F_TAND_idx = np.zeros((len(TAND_idx)
    #F_TAND_idx = TAND_idx[:int((TAND_idx.shape)*.3)

def reduce_clear(case):    
    
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to remove extra clear")
    
    #find where
    where = np.argwhere(case_4D['SM_Reflectance']<=20)
    print(where)
    #count the # points less than 20
    #COUNT1 = np.count_nonzero(np.argwhere((case_4D['SM_Reflectance'] <=20))
    #print(f"there are {COUNT1} datapoints less than 20%")
    
    ##randomly select to keep all GT20 and 10% lt=20
    # goes to another function
    #
    #
    #new_case_4D = np.nan_to_num(case_4D, copy=True, nan=9999, posinf=None, neginf=None)
    #filled = {"channels": channels}
    
    ###count the # points less than 20
    #COUNT2 = np.count_nonzero(np.isnan(new_case_4D))
    #print(f"there are now {COUNT2} data points less than 20%")
    
    
    #reassemble
    #print("reassembling the case")
    #for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
     #   filled[channels[i]] = new_case_4D[:,:,:,i] 
    #return filled
    
    
    

def main(args):
    #load
    print("im in remove cloud and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    # do the changes via a seperate fxn
    lessclear =reduce_clear(case)
    #save
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "_reduce_clear.npz"
    np.savez_compressed(savepath,**fillcase )
           