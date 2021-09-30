#attempt to run model

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

######THE MODEL######
def create_model(n_input): ####options
    #number of inputs ( decided before here)
    #activation functions ('relu', 'sigmoid', 'tanh')
    #lossfxn # can be mse or mae or a specialize function, MSE better for our problem

    # number of hidden layers

    # number of nodes per layer (if same one), can make different for each later
    n_units =4 

    model = tf.keras.Sequential()
    # First hidden layer:
    model.add(layers.Dense(n_units, activation='relu', input_shape=(n_input,) ))
    # Second hidden layer:
    model.add(layers.Dense(n_units, activation='relu'))
    # Output layer:  just 1 node and no activation function
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.01),  # Adam optimizer
                loss= 'mse',       # mean squared error
               metrics=['mae','mse'])  # mean absolute error
    return model