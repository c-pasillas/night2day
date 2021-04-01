###### IMPORTANT! ###
######
###### This code needs to be executed in a python environment that is set up
###### for innvestigate package.

import numpy as np
from matplotlib import pyplot as plt
import sys

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from keras import activations

from visualization_functions import *

# Self-defined functions
from load_data import load_data
#from plot_convergence import plot_convergence
from make_custom_file_names import model_file_name, data_file_name, heat_map_file_name_start
from custom_model_elements import my_r_square_metric
from prepare_data import ymax_default as ymax
from read_configuration import read_configuration
from default_configuration import defcon

### Read constants from files
#from constants_for_data_parameters import *
#from constants_for_normalization_parameters import *

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #catch FutureWarnings so that I can find actual errors!
    
    import keras
    import keras.backend as K
    import keras.models
    #import tensorflow as tf  # this is new for custom loss function
    from keras.models import load_model
    
    from keras.layers import Input, Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D, Conv2DTranspose, UpSampling2D, Reshape
    from keras.models import Model
    
    from keras.utils import CustomObjectScope
    from keras.initializers import glorot_uniform



##############################
# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()

##############################
# Choose which method to use
# 1 = Guided back propagation
# 2 = Gradient
# 3 = deconvnet
# 4 = LRP (with alpha = 1 or 2)
method_number = 4

# The following is only relevant for LRP.  Choose alpha=1 or alpha=2.
# How much positive attribution to use in explanation.
# alpha = 1:  only positive attribution allowed
# alpha = 2:  also some negative attribution allowed
alpha = 1  # How much positive attribution to use in explanation (alpha >= 1 required).

# Choose which pixel in output image we want to analyze.
my_row = 150  # 25
my_col = 100  # 120



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
modelfile = model_file_name(spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs )
data_file = data_file_name(spath)
heatmap_filestart = heat_map_file_name_start(spath,IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs)

######################################################################
################# DATA AND ANN MODEL ##########
# Step 1: loading data
print('\nLoading training and testing data')
Xdata_train, Ydata_train, Xdata_test, Ydata_test, Lat_train, Lon_train, Lat_test, Lon_test = load_data(data_file)
n_samples_testing,ny,nx,nchans = Xdata_test.shape

# Step 2: loading ANN model
print('Loading ANN model')
# This old version of Keras does not know the term 'GlorotUniform', so need to call it with an environemnt that defines it.
# Secondly - we choose NOT to compile the model after loading it.
# This means we can use the model for prediction, LRP, etc., but we cannot continue training it.
# With the non-compile option the model from TF2.0 is compatible with the old Keras, but otherwise it's not!
with CustomObjectScope({'GlorotUniform': glorot_uniform}):
    model = keras.models.load_model(modelfile,compile=False)

# Add a flatten layer to be able to prescribe individual neuron of interest in output layer.
model.add(Flatten())

print('\n')
print(model.summary())

##############################################
######## PREDICTIONS ##############

# Step 3: generate predictions
print('\nGenerate predictions')
Zdata_test = model.predict(Xdata_test)

# Do NOT restore original scaling, because we want to be able to match the pixel value at output to LRP values
# at input channels.  Can only do that, if we keep same scaling as during training.


###############################################
############## VISUALIZATION# #################
###############################################

# Step 4: Visualizatoion

### Last layer is already linear, no need to check.
model_wo_softmax = model

# my_neuron_selection_mode = "max_activation"  # This is the default method.  System chooses the pixel to analyze.
my_neuron_selection_mode = "index"  # This allows you to choose a specific pixel.

### Set up analyzers (method)

if method_number==1:
  # No arguments needed
  #my_analyzer = innvestigate.create_analyzer("guided_backprop", model_wo_softmax, neuron_selection_mode = "max_activation" )
  my_analyzer = innvestigate.create_analyzer("guided_backprop", model_wo_softmax, neuron_selection_mode = my_neuron_selection_mode )
  my_text = 'Guided back prop'
  method_text = 'Guided_back_prop'

elif method_number==2:
  # No arguments needed
  my_analyzer = innvestigate.create_analyzer("input_t_gradient", model_wo_softmax, neuron_selection_mode = my_neuron_selection_mode )
  my_text = 'Gradient'
  method_text = 'Gradient'

elif method_number==3:
  # No arguments needed
  my_analyzer = innvestigate.create_analyzer("deconvnet", model_wo_softmax, neuron_selection_mode = my_neuron_selection_mode )
  my_text = 'deconvnet'
  method_text = 'deconvnet'

elif method_number==4:
  if IS_UNET:
      print( 'LRP not implemented for UNET.')
      exit(1)

  if batchnorm:
      print( 'LRP not implemented for batchnorm')
      exit(1)

  if dropout:
      print( 'Caution - Scale of LRP results may be off when dropout is used.')
  # My favorite method.  LRP with alpha-beta rule.
  # Allows for both positive and negative attribution.
  # Play around with different values for alpha >= 1.
  # alpha = 1 or 2.   Defined at beginning of file.
  # How much positive attribution to use in explanation (alpha >= 1 required).
  beta = alpha-1  # How much negative attribution to use in explanation.
  bias = True # Does not matter much for now.  Ignore for now.
  my_analyzer = innvestigate.create_analyzer("lrp.alpha_beta", model_wo_softmax, alpha=alpha, beta=beta, bias=bias, neuron_selection_mode = my_neuron_selection_mode )

  # Explain what we're seeing with this analyzer.
  my_text = 'LRP alpha-beta rule, alpha={} beta={}.'.format(alpha,beta)
  method_text = 'LRP_alpha_' + repr(alpha)

  print('Using LRP with alpha-beta rule, alpha={} beta={}.'.format(alpha,beta))
  print('Alpha = portion of positive attribution allowed (in red), beta = portion of negative attribution allowed (in blue).\n')

  # Alternative way to define the above analyzer:
  # lrp_analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta(model_wo_softmax, alpha=alpha, beta=beta, bias=bias )
  # List of all available methods can be found here:  https://innvestigate.readthedocs.io/en/latest/modules/analyzer.html


###############################################
### Step 4: Get the samples you want to analyze and choose pixel to analyze
i_testsample = 0
my_sample = Xdata_test[i_testsample,:,:,:]
n_samples_to_analyze = 1
# Get the sample
my_sample = my_sample.reshape(n_samples_to_analyze,nx,ny,nchans)
# ANN prediction for that sample
my_estimate = Zdata_test[i_testsample,:]


# Transform pixel location (my_row, my_col) into single index:
my_index = my_row * ny + my_col   # Should this be nx or ny?  Doesn't matter here, because nx=ny.
analysis = my_analyzer.analyze(my_sample,neuron_selection=my_index)

###############################################
### Step 5: Perform the actual analysis for the samples.  Get heatmap!
### Visualize results.

### Build in this safety check:
pixel_value = my_estimate[my_index]
if (pixel_value <= 0 and method_number == 4):
    print( '\n --- Warning - LRP not meaningful for this pixel! --- \n')
    print(' --- Reason: estimated output at pixel is not positive: {}. ---'.format(pixel_value))
    exit(1)

n_rows = 2
n_cols = nchans
f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))

# Add description for each of the channels - this is only used if there are indeed four channels.
my_desription_list = [ 'GOES ABI #1', 'GOES ABI #2', 'GOES ABI #3', 'GOES GLM'] #, 'MRMS estimate', 'MRMS actual']

# find maximum value for entire array
heatmap_z_abs_max, heatmap_z_min, heatmap_z_max, heatmap_non_zero_values_exist = find_symmetric_range_for_colormap( analysis, my_epsilon )

for i_chan in range(nchans):

   my_image = my_sample[:,:,:,i_chan].reshape(ny,nx)
   input_z_abs_max, input_z_min, input_z_max, input_non_zero_values_exist = find_symmetric_range_for_colormap( my_image, my_epsilon )
   axes[0,i_chan].imshow(my_image, cmap='RdGy', clim=(-input_z_abs_max,input_z_abs_max) )
   if (nchans == 4):
       this_text = my_desription_list[i_chan]   # If there are 4 input channels, we know what they are and can label them.
   else:
       this_text = ''
   axes[0,i_chan].set_title('{}  [{:#.3F},{:#.3F}]'.format(this_text,input_z_min,input_z_max) )
   # Add circle in feature map at location of interest:
   my_circle = plt.Circle((my_col, my_row), 6, color='red', fill=False)
   axes[0, i_chan].add_artist(my_circle)

   my_image = analysis[:,:,:,i_chan].reshape(ny,nx)
   channel_sum = np.sum( my_image )
   # Note that we use the same scale across all channels!
   # Important to see relative contribution of each channel.
   axes[1,i_chan].imshow(my_image,cmap='seismic', clim=(-heatmap_z_abs_max,heatmap_z_abs_max))
   #axes[1,i_chan].set_title('{} [{:#.3F},{:#.3F}] SUM={:#.3F}'.format(this_text,heatmap_z_min,heatmap_z_max,channel_contribution) )
   axes[1,i_chan].set_title('{}  SUM={:#.5F}'.format(this_text, channel_sum) )

f.suptitle('Heatmap for row={} col={}.\nPixel value: {:#.5F}\nMethod: {}'.format(my_row, my_col, pixel_value, my_text) )

my_filename = heatmap_filestart + '_Innvestigate_row_' + repr(my_row) + '_col_' + repr(my_col) + '_' + method_text + '.png'
print('Saving file to ' + my_filename )
f.savefig(my_filename)
plt.close(f)







