#attempt to run model from already saved TORS/TANDS
#FNN retrain
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

def FNN_retrain(args):
    print("im in FNN retrain")
    case = np.load(args.npz_path)
    print("I loaded the case")
    print("I am loading TORS_t, TORS_v, TAND_t, TAND_v")
   
    tors_t = case['tors_t']
    tors_v = case['tors_v']
    tand_t = case['tand_t']
    tand_v = case['tand_v']
    print("shape of TORS_t TAND_t TORS_v TANDS_v arrays are:", tors_t.shape, tand_t.shape, tors_v.shape, tand_v.shape)
    #reshape for model use
    TORS_t = tors_t.reshape (-1,tors_t.shape[-1]) 
    TORS_v = tors_v.reshape(-1, tors_v.shape[-1]) 
    TAND_t = tand_t.flatten()
    TAND_v = tand_v.flatten()
    print("shape of TORS_t TAND_t TORS_v TANDS_v arrays are:", TORS_t.shape, TAND_t.shape, TORS_v.shape, TAND_v.shape)
    
    print("i am now making the model")
    n_input = len(TORS_t)
    model = create_model(n_input)
                         
    # number of epochs to train 
    n_epochs = 2
    #batch size, # patches before update small=finer resolution/> time may get in a minumum, large < time may jump a minimum
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
    print("done with model training")