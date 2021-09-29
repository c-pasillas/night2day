#import statements
import numpy as np
import gzip
import math
import DNB_norm
import pathlib
import aoi
import patch
import NAN
import Mband_norm
import DNB_norm
import VIIRS_btd
import I2M_all
from common import log

#inputs are colocated case, predictors and predictand

##### FNN_train main
def maintrain(args):
    #make the training case
    log.info(f"I loaded the case and arguments are {args.train_path},{args.val_path}, {args.aoi}, {args.Predictors}") 
    print("im in FNN train and my args are", args)
    traincase = np.load(args.train_path)
    print(" I am making the training inputs")
    traincase2 = I2M_all.I2M_all_case(traincase)
    TORS_t, TAND_t = set_up(traincase2, predictors,predictand)
    
    #make the validation inputs TORS_v, TAND_v
    #with np.load('args.val_path') as valcase:
    valcase = np.load(args.val_path)
    print(" I am making the validation inputs")
    valcase2 = I2M_all.I2M_all_case(valcase)
    TORS_v, TAND_v = set_up(valcase2, predictors,predictand)
    
    #make the model
    print("i am now making the model")
    n_input = len(args.Predictors)
    model = create_model(n_input)
    # number of epochs to train 
    n_epochs = 2
    #batch size, # patches before update small=finer resolution/> time may get in a minumum, large < time may jump a minimum
    bs = 1000
    history = model.fit(TORS_t, TAND_t,validation_data =(TORS_v,TAND_v), epochs=n_epochs, batch_size=bs)

    print("I am now saving the model")
    made =time.strftime("%Y-%m-%dT%H%M")
    model.save(f'FNN_{made}')

    #save the history as a text file
    with open (f"myhistory_{made}", "w") as f:
        import pprint
        pprint.pprint(history.history, stream =f)
    #save the channels
    write_channels(f"model_channels_{made}", args.DNB, args.Predictors)###
    #save the tors/tand arrays for retraining
    print(" i am saving the tors and tands for future use")
    npsavez(f"TORTANDS_{made}", tors,tand)



###HELPER FUNCTIONS##############
########MAKE TORS/TANDS
def set_up(case, predictors, predictand):
    tors = np.stack([case[c] for c in predictors], axis = -1) #X in array
    tand = case[predictand] #y in array, 

    TORS = tors.reshape(-1, len(predictors))
    TAND = tand.flatten()

    return (TORS, TAND)

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



