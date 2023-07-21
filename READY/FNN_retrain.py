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
from common import log, rgb, reset, blue, orange, yellow, bold

###helper fxn

def rewrite_channels(path, inputfile, n_epocs, bs):
    with open (path, "w") as f:
        print(inputfile, file =f)
        print(n_epocs, file =f)
        print(bs, file = f)
        print(f"go to file to see predictors and predictand", file =f)
        
        #for p in TORS:
         #   print(p, file = f, end = "\n")   
        print(file =f)




######THE MODEL######
def create_model(n_input): ####options
    #number of inputs ( decided before here)
    #activation functions ('relu', 'sigmoid', 'tanh')
    #lossfxn # can be mse or mae or a specialize function, MSE better for our problem
    # number of hidden layers
    # number of nodes per layer (if same one), can make different for each later
    n_units =8 

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
    
    
    #print("I am loading TORS_t, TORS_v, TAND_t, TAND_v")
   
    #tors_t = case['tors_t']
    #tors_v = case['tors_v']
    #tand_t = case['tand_t']
    #tand_v = case['tand_v']
    #print("shape of TORS_t TAND_t TORS_v TANDS_v arrays are:", tors_t.shape, tand_t.shape, tors_v.shape, tand_v.shape)
    #reshape for model use
    #TORS_t = tors_t.reshape (-1,tors_t.shape[-1]) 
    #TORS_v = tors_v.reshape(-1, tors_v.shape[-1]) 
    #TAND_t = tand_t.flatten()
    #TAND_v = tand_v.flatten()
    #print("shape of TORS_t TAND_t TORS_v TANDS_v arrays are:", TORS_t.shape, TAND_t.shape, TORS_v.shape, TAND_v.shape)
    print("i am prepping data for analysis")
    ts = 0.2
    #random seed
    rs = 10
    TORS = case['arr_0']
    TAND = case['arr_1']
    
    print("i am starting train_test_split")
    TORS_train, TORS_test, TAND_train, TAND_test  = train_test_split(TORS, TAND, test_size=ts, random_state=rs)
    
    #TORS_t = tors_t.reshape (-1,tors_t.shape[-1]) 
    #TORS_v = tors_v.reshape(-1, tors_v.shape[-1]) 
    #TAND_t = tand_t.flatten()
    #TAND_v = tand_v.flatten()
    #print("shape of TORS_t TAND_t TORS_v TANDS_v arrays are:", TORS_t.shape, TAND_t.shape, TORS_v.shape, TAND_v.shape)
    
    print("i am now making the model")
    n_input = 10
    model = create_model(n_input)
    
    
    # number of epochs to train 
    n_epochs = 25
    #batch size, # patches before update small=finer resolution/> time may get in a minumum, large < time may jump a minimum
    bs = 2500
    history = model.fit(TORS_train, TAND_train,validation_data =(TORS_test,TAND_test), epochs=n_epochs, batch_size=bs)

    log.info(f'I am now saving the model')
    made =time.strftime("%Y-%m-%dT%H%M")
    model.save(f'M{made}_FNNretrain_{args.npz_path[:-4]}')
    log.info(f'I saved the model')
    
    
     #save the history as a text file
    log.info(f' i am saving the history as a text file')
    with open (f"M{made}_myhistory", "w") as f:
        import pprint
        pprint.pprint(history.history, stream =f)
    #save the channels 
    log.info(f' i am saving the channels')
    rewrite_channels(f"M{made}_FNNretrain_model_channels_{args.npz_path[:-4]}", args.npz_path, n_epochs, bs)
    
     #print chart of training info
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_mae'], label = 'validation_mae')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.ylim([0, .1])
    plt.xlim([0,30])
    plt.legend(loc='lower right')
    plt.savefig(f"M{made}_trainingstats.png") 
    plt.clf()
    plt.close()
    
    metrics_df = pd.DataFrame(history.history)
    
    metrics_df[['loss', 'val_loss']].plot().get_figure().savefig('M{made}_loss.png')
    
    metrics_df[['mae', 'val_mae']].plot().get_figure().savefig('M{made}_mae.png')
    
    metrics_df[['mse', 'val_mse']].plot().get_figure().savefig('M{made}_mse.png')
    
    log.info(f'done with model training')
    