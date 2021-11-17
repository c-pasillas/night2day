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
    tand =  (np.clip(tand, 0, 100)) / 100 #adjsutes for any over 100% reflectance (ie  city lights) and also makes 0-1 vs 0-100
    
    TORS = tors.reshape(-1, len(PREDICTORS))
    TAND = tand.flatten()
    #print("tands are", tand(:,0:10))
    #print("tors are", tors(:,0:10,:))
    #print("TORS are" + TORS(:,0:10))
    #print("TANDS are" + TAND(:,0:10))
    print(TORS.shape)
    print(TAND.shape)
    #split the data to test train
    #training split
    ts = 0.2
    #random seed
    rs = 10
    
    return(TORS, TAND,ts, rs)
    #return train_test_split(TORS, TAND, test_size=ts, random_state=rs)


######THE MODEL######
######THE MODEL######
def create_model(n_input): ####options
    #number of inputs ( decided before here)
    #activation functions ('relu', 'sigmoid', 'tanh')
    #lossfxn # can be mse or mae or a specialize function, MSE better for our problem

    # number of hidden layers

    # number of nodes per layer (if same one), can make different for each later
    n_units =2 

    model = tf.keras.Sequential()
    # First hidden layer:
    model.add(layers.Dense(n_units, activation='relu', input_shape=(n_input,) ))
    # Second hidden layer:
    model.add(layers.Dense(n_units, activation='relu'))
    # Output layer:  just 1 node and no activation function
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.summary()
    model.compile(optimizer='adam',loss= 'mse', metrics=['mae','mse'])  # mean absolute error
    
    #model.compile(optimizer=keras.optimizers.Adam(0.01),  # Adam optimizer
     #           loss= 'mse',       # mean squared error
      #         metrics=['mae','mse'])  # mean absolute error
    
    return model

def FNN_train(args):
    print("im in FNN train and my args are", args)
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
    log.info(f'I am now making the model')
    model = create_model(n_input)
                         
    #history = model.fit(x, y, validation_split=0.30, epochs=n_epochs, batch_size=128)
    # number of epochs to train 
    n_epochs = 1
    #batch size, # patches before update small=finer resolution/> time may get in a minumum, large < time may jump a minimum
    bs = 1000
    history = model.fit(TORS_train, TAND_train,validation_data =(TORS_test,TAND_test), epochs=n_epochs, batch_size=bs)

    log.info(f'I am now saving the model')
    made =time.strftime("%Y-%m-%dT%H%M")
    model.save(f'FNN_{args.npz_path[:11]}_{made}')
    log.info(f'I saved the model')

    #save the history as a text file
    log.info(f' i am saving the history as a text file')
    with open (f"myhistory_{made}", "w") as f:
        import pprint
        pprint.pprint(history.history, stream =f)
    #save the channels
    log.info(f' i am saving the channels')
    write_channels(f"FNN_model_channels_{made}", args.npz_path, PREDICTORS, PREDICTAND)
    log.info(f' i am saving the tors/tands for retrain')
    savepath = args.npz_path[:-4]+ f"TORS_TAND.npz"
    np.savez_compressed(savepath, TORS, TAND)
    log.info(f'done with model training')
  
    
