import numpy as np
from matplotlib import pyplot as plt
import sys

# Self-defined functions
from load_data import load_data
from plot_convergence import plot_convergence
from refc_statistics import get_refc_stats
from make_custom_file_names import model_file_name
from make_custom_file_names import history_file_name
from make_custom_file_names import data_file_name
from make_custom_file_names import predictions_file_name
from make_custom_file_names import convergence_plot_file_name
from custom_model_elements import my_r_square_metric
from prepare_data import ymax_default as ymax
from read_configuration import read_configuration
from default_configuration import defcon

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    from tensorflow.keras.models import load_model

# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()

##############################
# Choose whether to do the following
WANT_SAVE_PREDICTIONS = True
WANT_PRINT_STATISTICS = True
WANT_CONVERGENCE_PLOTS = True

##############################
### Choose model for which to run the visualization - by parameters:

spath = '..'  #Imme

try:
    machine = config['machine']
except KeyError:
    try:
        machine = config[my_file_prefix]['machine']
    except KeyError:
        machine = defcon['machine']
print('machine =',machine)

if machine == 'Hera':
    spath = '/scratch1/RDARCH/rda-goesstf/conus2'  #KH on Hera

try:
    data_suffix = config['data_suffix']
except KeyError:
    try:
        data_suffix = config[my_file_prefix]['data_suffix']
    except KeyError:
        data_suffix = defcon['data_suffix']
print('data_suffix =',data_suffix)

try:
    NN_string = config['NN_string']
except KeyError:
    try:
        NN_string = config[my_file_prefix]['NN_string']
    except KeyError:
        NN_string = defcon['NN_string']
print('NN_string =',NN_string)

if NN_string == 'SEQ':
    IS_UNET = False
else:
    IS_UNET = True
print('IS_UNET =',IS_UNET)

try:
    n_encoder_decoder_layers = config['n_encoder_decoder_layers']
except KeyError:
    try:
        n_encoder_decoder_layers = config[my_file_prefix]['n_encoder_decoder_layers']
    except KeyError:
        n_encoder_decoder_layers = defcon['n_encoder_decoder_layers']
print('n_encoder_decoder_layers =',n_encoder_decoder_layers)

try:
    nepochs = config['nepochs']
except KeyError:
    try:
        nepochs = config[my_file_prefix]['nepochs']
    except KeyError:
        nepochs = defcon['nepochs']
print('nepochs =',nepochs)

######################################################################
### Filenames for model, history, data, etc.

modelfile = model_file_name(spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs )
historyfile = history_file_name( spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs )
data_file = data_file_name( spath, suffix=data_suffix ) # load file name from file
predictionsfile = predictions_file_name(spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs)
convergence_plot_file = convergence_plot_file_name(spath, IS_UNET, \
    my_file_prefix, n_encoder_decoder_layers, nepochs)

######################################################################
################# DATA AND ANN MODEL ##########
# Step 1: loading data
print('\nLoading training and testing data')
print('loading data from file =',data_file)
Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
    Lat_train, Lon_train, Lat_test, Lon_test = load_data(data_file)

n_samples_training_real = Xdata_train.shape[0]
n_samples_testing = Xdata_test.shape[0]

# Step 2: loading ANN model
print('Loading ANN model: ' + modelfile )
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    model = load_model(modelfile, \
        custom_objects={"my_r_square_metric": my_r_square_metric})
    print('\n')
    print(model.summary())
    print('\n')

######################################################################
######## PREDICTIONS ##############
# Step 3: generate predictions
print('\nGenerate predictions for training and test data')
print('Loading ANN model: ' + modelfile )
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    Zdata_train = model.predict(Xdata_train)
    Zdata_test = model.predict(Xdata_test)

# Step 4: RESTORE ORIGINAL SCALING ###
print('\nRestore original scaling')
Zdata_train = np.array(Zdata_train,dtype=np.float64)
Zdata_test = np.array(Zdata_test,dtype=np.float64)

# note: ymin is zero
Zdata_train *= ymax
Ydata_train *= ymax
Zdata_test *= ymax
Ydata_test *= ymax

print('Zdata_train min,mean,max=',\
    np.min(Zdata_train),np.mean(Zdata_train),np.max(Zdata_train))
print('Ydata_train min,mean,max=',\
    np.min(Ydata_train),np.mean(Ydata_train),np.max(Ydata_train))
print('Zdata_test min,mean,max=',\
    np.min(Zdata_test),np.mean(Zdata_test),np.max(Zdata_test))
print('Ydata_test min,mean,max=',\
    np.min(Ydata_test),np.mean(Ydata_test),np.max(Ydata_test))


###############################################
############## OPTIONAL ITEMS #################
###############################################

###################################
# Generate convergence plots
if WANT_CONVERGENCE_PLOTS:
   print('Convergence plots')
   plot_convergence(historyfile,convergence_plot_file)
   
###################################
# SAVE RESTORED PREDICTIONS TO FILE ###
if WANT_SAVE_PREDICTIONS:
   print('Saving predictions to file: ' + predictionsfile)
   f = open(predictionsfile,'wb') # open file
   Ydata_train[0:n_samples_training_real,:,:].tofile(f) # truth-training
   Zdata_train[0:n_samples_training_real,:,:].tofile(f) # estimate-training
   Ydata_test.tofile(f)  # truth - testing
   Zdata_test.tofile(f)  # estimate - testing
   Lat_train.tofile(f) # lat-lon coordinates
   Lon_train.tofile(f)
   Lat_test.tofile(f)
   Lon_test.tofile(f)
   f.close()  # end of file

###################################
### PRINT STATISTICS ON SCREEN ###
if WANT_PRINT_STATISTICS:

   print('\nDimensions of data types')
   # First print dimensions of results
   print(Ydata_train.dtype,Ydata_train[0:n_samples_training_real,:,:].shape)
   print(Zdata_train.dtype,Zdata_train[0:n_samples_training_real,:,:].shape)
   print(Ydata_test.dtype,Ydata_test.shape)
   print(Zdata_test.dtype,Zdata_test.shape)
   print(Lat_train.dtype,Lat_train.shape)
   print(Lon_train.dtype,Lon_train.shape)
   print(Lat_test.dtype,Lat_test.shape)
   print(Lon_test.dtype,Lon_test.shape)

   ################################################################
   # Print result for each original sample (not for augmentated cases)
   
   # Kyle - Seems to me as if this goes through the first 224 samples,
   # including augmented samples, adn skis the rest.
   # Maybe use i_training_sample * (augfac+1) below?
   
   print('\nStatistics of results - Training data')
   print('Sample  rmsd   rsq   csi3   csi6')
   for i_training_sample in range(n_samples_training_real):
      zdata = Zdata_train[i_training_sample,:,:]
      ydata = Ydata_train[i_training_sample,:,:]

      #stats = check_goes_refc(zdata,ydata)
      stats = get_refc_stats(zdata,ydata)

      print(i_training_sample,\
        '{0:8.4f}'.format(stats['rmsd']),\
        '{0:6.4f}'.format(stats['rsq']),\
        '{0:6.4f}'.format(stats['csi'][3]),\
        '{0:6.4f}'.format(stats['csi'][6]),\
        )

   print('\nStatistics of results - Test data')
   print('Sample  rmsd   rsq   csi3   csi6')
   for i_test_sample in range(n_samples_testing):

      zdata = Zdata_test[i_test_sample,:,:]
      ydata = Ydata_test[i_test_sample,:,:]

      #stats = check_goes_refc(zdata,ydata)
      stats = get_refc_stats(zdata,ydata)

      print(i_test_sample,\
        '{0:8.4f}'.format(stats['rmsd']),\
        '{0:6.4f}'.format(stats['rsq']),\
        '{0:6.4f}'.format(stats['csi'][3]),\
        '{0:6.4f}'.format(stats['csi'][6]),\
        )
     
################################################################
