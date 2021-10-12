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
import DNB_norm
import pathlib
import FNN_assess
import FNN_predict

    
###
def both(args):
    
    ZZ = FNN_predict(args.)
    FNN_assess(ZZ)
    print("i am done running assessment, look in folders for data")