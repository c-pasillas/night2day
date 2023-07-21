#attempt asses the results of the model #needs ALOT OFWORK

#import statements
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
import post_helpers as ph
import pathlib


def load_data(truth, ML):
    #assign variables
    SM_REFLECTANCE = truth['SM_reflectance']
    ML_REFLECTANCE = ML['MLtruth']
    REFLECT_DIFF = SM_REFLECTANCE - ML_REFLECTANCE
    
    #need to also add the actuall raw channels for the channel images
    
    normML = ML['MLtruth']
    PREDICTAND_LABEL = ML["channel"][0]
    #Reflect_diff
    
    x = SM_REFLECTANCE
    x_clip = (np.clip(x, 0, 100)) / 100
    y = ML_REFLECTANCE
    fixdiff = x_clip - y
    print('done with load-data')
    return {"x": SM_REFLECTANCE, "x_clip": x_clip, "y": ML_REFLECTANCE, "fixdiff":fixdiff} #, "M13": M13, "M14": M14, "M15": M15, "M16": M16,  }
    
    
    
    
    
def run_stats(data_dic, name):   
    # basic stats
    print("starting basic stats")
    ph.basic_stats(data_dic, name)
    
    print("starting xy relations for Reflectances")
    ph.xy_relations(data_dic["x"],data_dic["y"])
   
    print("starting xy relations for clipReflectances")
    ph.xy_relations(data_dic["x_clip"],data_dic["y"])
    
    print("making the PDFs")
    ph.plotdata(data_dic['x_clip'], data_dic['y'])#, channel_name, figdir, nick):
    
    #print("i am making plots")
    #ph.plotit(data_dic["x_clip"],data_dic["y"])
    #ph.plotit(data_dic["x_clip"][0:5],data_dic["y"][0:5])
    #print("draw cole")
    #ph.draw_COLE(data_dic, name)
    
    #print("making value diff plots")
    #ph.colordiff(data_dic, name)
 
    #print("making the PDFs")
    #ph.plotdata(data_dic['x_clip'], data_dic['y'])#, channel_name, figdir, nick):
    
    print("making x clip hexplot")
    ph.hex_plt(data_dic["x_clip"], data_dic["y"])
    
    print("making x clip hexyplot")
    ph.hexy_plt(data_dic["x_clip"], data_dic["y"])
    
    #print("making the PDFs")
    #ph.plotdata(data_dic['x_clip'], data_dic['y'])#, channel_name, figdir, nick):
    
    #print('making density scatter plot")
    #ph.density_scatter( data_dic['x'],  data_dic['y'], bins = [30,30] )
    #print("making ERF plots and calcs")
    #ph.ERF(data_dic["x"],data_dic["y"])


#############
def assessment(args):
    truth = np.load(args.npz_path)
    name =pathlib.Path(args.npz_path).resolve().stem
    print("I loaded the DNB/Channels case")
    ML = np.load(args.ML_path)
    print("I loaded the ML_DNB case")
    
    data_dic = load_data(truth, ML)#currently a dict of arrays 
    print('Done with load_data(truth, ML)')
    analysis = run_stats(data_dic, name)
    print("i am done running assessment, look in folders for data")
