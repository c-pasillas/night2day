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
import patch
import NAN

def write_channels(path, TAND, TORS):
    with open (path, "w") as f:
        print(TAND, file = f)
        for p in TORS:
            print(p, file = f, end = " ")
        print(file =f)

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
    #lossfxn # can be mse or mae or a specilaize function, MSE better for our problem

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

    #heres the model 
    return model

def FNN_train(args):
    print("im in FNN train and my args are", args)
    case = np.load(args.npz_path)
    print("I loaded the case") #case is large 3K4K
    case = patch(case) # patch the case
    case = NAN9999(case) # remove patches with 9999
    if args.aoi:
        case = aoi.aoi_case(case, args.aoi)
    print(" I am making the inputs")
    TORS_train, TORS_test, TAND_train, TAND_test  = set_up(case, args.Predictors, args.DNB)
    print("i am now making the model")
    n_input = len(args.Predictors)
    model = create_model(n_input)
    #history = model.fit(x, y, validation_split=0.30, epochs=n_epochs, batch_size=128)
    # number of epochs to train 
    n_epochs = 2
    #batch size, # patches before updates smaller finer resolution/ > time may get in a minumum, larger < time may miss over a minimum
    bs = 1000
    history = model.fit(TORS_train, TAND_train,validation_data =(TORS_test,TAND_test), epochs=n_epochs, batch_size=bs)   
    print("I am now saving the model")
    made =time.strftime("%Y-%m-%dT%H%M")
    model.save(f'FNN_{made}')
    #save the history as a text file
    with open (f"myhistory_{made}", "w") as f:
        import pprint
        pprint.pprint(history.history, stream =f)
    #save the channels
    write_channels(f"model_channels_{made}", args.DNB, args.Predictors)
    print("done with model training")
    
           