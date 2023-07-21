#attempt to run MLR  model

#import statements

import tensorflow as tf
from tensorflow import keras
import gzip
import math
import tensorflow.keras.layers as layers
import IPython.display as dis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
from sklearn import linear_model
from sklearn import metrics
import os
import sys
import seaborn as sb
from matplotlib import style
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
scaler = StandardScaler()
from common import log
from pathlib import Path
from datetime import datetime
from PIL import Image
import pprint





#command for the GPU
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#load data
case=np.load('TESTsmall_MODELready.npz')
case.files

TORS = case['TORS']
TAND = case['TAND']
o_shape = TAND.shape


#load the  MLR model that was created and evaluate it on a data set
model = tf.keras.models.load_model('regOLS')

with open (model_path, 'rb') as g:
        model = pickle.load(g)

#lets predict on all the data
model_output= model.predict(TORS)

model_outputfinal= model_output.reshape(o_shape)
model_outputfinal.shape

#save the ML truths
savepath = args.npz_path[:9]+ "_MLtruth.npz"
np.savez(savepath, MLtruth = model_outputfinal)
