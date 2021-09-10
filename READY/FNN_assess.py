#attempt asses the results of the model #needs ALOT OFWORK

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
import post_helpers as ph
import DNB_norm
import pathlib


def load_data(truth, ML):
    #assign variables #may need to also do the original channels here since plan to compare to them visually
    M12 = truth["M12"]
    M13 = truth["M13"]
    M14 = truth["M14"]
    M15 = truth["M15"]
    M16 = truth["M16"]
    DNB = truth['DNB']
    MLnorm = ML['MLtruth']
    PREDICTAND_LABEL = ML["channel"][0]
    ML_DNB = DNB_norm.denormalize(PREDICTAND_LABEL, MLnorm)

    DNBdiff = DNB - ML_DNB
    x = DNB
    y = ML_DNB
    print('done with load-data')
    return {"M12": M12, "M13": M13, "M14": M14, "M15": M15, "M16": M16, "DNB": DNB, "DNBdiff": DNBdiff, "x": DNB, "y": ML_DNB}
    
def run_stats(data_dic, name):   
    #basic stats
    print("starting basic stats")
    ph.basic_stats(data_dic, name)
    print("starting xy relations")
    ph.xy_relations(data_dic["x"],data_dic["y"])
    print("draw cole")
    ph.draw_COLE(data_dic, name)
    #scatter/densityplot DNB vs ML_DNB and save in folder with ML truths

#############
def assessment(args):
    truth = np.load(args.npz_path)
    name =pathlib.Path(args.npz_path).resolve().stem
    print("I loaded the DNB/Channels case")
    ML = np.load(args.ML_path)
    print("I loaded the ML_DNB case")
    
    data_dic = load_data(truth, ML)  #currently a dict of arrays 
    analysis = run_stats(data_dic, name)
    print("i am done running assessment, look in folders for data")
