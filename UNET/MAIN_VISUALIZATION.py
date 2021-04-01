import numpy as np
from matplotlib import pyplot as plt
import sys

# Self-defined functions
from load_data import load_data
from plot_convergence import plot_convergence
from make_custom_file_names import \
    model_file_name, \
    history_file_name, \
    data_file_name, \
    convergence_plot_file_name, \
    prediction_plot_file_name_start, \
    feature_map_file_name_start, \
    backward_optimization_file_name_start, \
    input_patches_file_name_start
from custom_model_elements import my_r_square_metric
from prepare_data import ymax_default as ymax
from read_configuration import read_configuration
from default_configuration import defcon
from visualization_functions import *
    #\
    #find_symmetric_range_for_colormap,\
    #calc_and_plot_backward_optimization_results_one_layer,\
    #calc_and_plot_backward_optimization_history,\
    #calc_and_plot_feature_maps,\
    #calc_and_plot_MAX_INPUT_PATCHES_one_layer_one_feature_selected_samples,\
    #calc_and_plot_MAX_INPUT_PATCHES_one_layer_one_sample_all_features

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    from tensorflow.keras import models
    from tensorflow.keras.models import load_model

# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()

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

#####

# Need to run predictions on test data.  Do we also want predictions for training data?  Currently no needed.
ALSO_PREDICT_TRAINING_DATA = False
WANT_CONVERGENCE_PLOT = True #True
WANT_PREDICTION_PLOTS = False #True
WANT_BACKWARD_OPTIMIZATION_MAPS = True
WANT_FEATURE_MAPS = False # True
# The following only works if WANT_BACKWARD_OPTIMIZATION_MAPS is True as well.
# Reason:  We use backward optimization to estimate the ERF for each layer, which is needed for this.
WANT_FEATURE_ACTIVATION_INPUT_PATCHES = True

# initialize lists for effective receptive field
layer_indices_for_ERF = []
ERF_list = []

######################################################################
### Filenames for model, history, data, etc.
modelfile = model_file_name(\
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs )
historyfile = history_file_name( \
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs )
#data_file = data_file_name( )
data_file = data_file_name( spath, suffix=data_suffix ) # load file name from file
prediction_plot_file_start = prediction_plot_file_name_start( \
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs)
feature_map_file_start = feature_map_file_name_start(\
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs)
convergence_plot_file = convergence_plot_file_name(\
    spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs)
backward_optimization_file_start = backward_optimization_file_name_start(spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs)
input_patches_file_start = input_patches_file_name_start(spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs)

######################################################################
################# DATA AND ANN MODEL ##########
# Step 1: loading data
print('\nLoading training and testing data')
print('loading data from file =',data_file)
Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
    Lat_train, Lon_train, Lat_test, Lon_test = load_data(data_file)

n_samples_testing,ny,nx,nchans = Xdata_test.shape
print('n_samples_testing=',n_samples_testing)

# Step 2: loading ANN model
print('Loading ANN model')
# For any model with custom objects (here custom metric), need to provide dictionary that maps
# function name to actual function.
model = load_model(modelfile, \
    custom_objects={"my_r_square_metric": my_r_square_metric})
print('\n')
print(model.summary())
print('\n')

######################################################################
######## PREDICTIONS ##############
# Step 3: generate predictions
print('\nGenerate predictions')
if ALSO_PREDICT_TRAINING_DATA:
   Zdata_train = model.predict(Xdata_train)

Zdata_test = model.predict(Xdata_test)

# Step 4: RESTORE ORIGINAL SCALING - FOR OUTPUT CHANNEL ONLY ###
print('\nRestore original scaling')
if ALSO_PREDICT_TRAINING_DATA:
    Zdata_train = np.array(Zdata_train,dtype=np.float64)
    Zdata_train *= ymax
    Ydata_train *= ymax
    print('Zdata_train min,mean,max=', \
        np.min(Zdata_train), np.mean(Zdata_train), np.max(Zdata_train))
    print('Ydata_train min,mean,max=', \
        np.min(Ydata_train), np.mean(Ydata_train), np.max(Ydata_train))

Zdata_test = np.array(Zdata_test,dtype=np.float64)
Zdata_test *= ymax
Ydata_test *= ymax
print('Zdata_test min,mean,max=',\
    np.min(Zdata_test),np.mean(Zdata_test),np.max(Zdata_test))
print('Ydata_test min,mean,max=',\
    np.min(Ydata_test),np.mean(Ydata_test),np.max(Ydata_test))

# Done with preparation steps.

###############################################
############## VISUALIZATION# #################
###############################################

############### Visualization Task #1 ###############
# Generate convergence plot
if WANT_CONVERGENCE_PLOT:
    print('Convergence plots')
    plot_convergence(historyfile,convergence_plot_file)

############### Visualization Task #2 ###############
# Visualize some samples: input and output


if WANT_PREDICTION_PLOTS:

    print('Generating predictions for all test samples and save to file (one file per sample)')
    
    for i_sample in range( n_samples_testing ):
    
        print('Test sample ' + repr(i_sample))

        f, axes = plt.subplots(1, 6, figsize=(20, 8))
        
        # Assemble list of 6 images to show
        my_image_list = [ Xdata_test[i_sample,:,:,0], Xdata_test[i_sample,:,:,1], Xdata_test[i_sample,:,:,2], Xdata_test[i_sample,:,:,3], Zdata_test[i_sample,:,:], Ydata_test[i_sample,:,:] ]
        # Add description for each of the 6 images
        my_desription_list = [ 'GOES ABI #1', 'GOES ABI #2', 'GOES ABI #3', 'GOES GLM', 'MRMS estimate', 'MRMS actual']
        
        for i in range(6):
            my_image = my_image_list[i].reshape(ny,nx)
            # Create color scale that uses symmetric range around 0, i.e. of the form [-M,M].
            z_abs_max, z_min, z_max, non_zero_values_exist = find_symmetric_range_for_colormap( my_image, 0.0001 )
            #axes[i].imshow(my_image,cmap='RdGy',clim=(-z_abs_max,z_abs_max))
            axes[i].imshow(my_image,cmap='bone', clim=(0,z_abs_max) )
            axes[i].set_title('{}  [{:#.1F} , {:#.1F}]'.format(my_desription_list[i],z_min,z_max) )
        
        my_plot_filename = prediction_plot_file_start + '_testsample_' + repr(i_sample) + '.png'

        print('Saving sample plot to file ' + repr(my_plot_filename) )
        plt.savefig( my_plot_filename )
        plt.close()


############### Visualization Task 3 ###############
# Feature visualization - aka backward optimization
# 1) Take an input: random / all zero / existing input sample
# 2) Modify input to maximize activation level of a specific neuron,
#    e.g., the center pixel of a feature map - which represents
#    one location of a filter activation.


if WANT_BACKWARD_OPTIMIZATION_MAPS:

    print( '\nVisualize intermediate neurons using BACKWARD OPTIMIZATION\n   and calculate Effective Receptive Field (ERF)\n' )

    # Choose parameters
    
    # Step 1: Choose loss function to use
    #i_loss_function = 0 # maximize activation mean of feature across all locations in feature map
    i_loss_function = 1 # maximize activation of filter at only one location in feature map (center)
    
    # Step 2: Choose step size and number of iterations
    grad_step_size = 0.2  # step size for gradient ascent
    n_iterations = 20   # How many steps
    save_every_n_th_figure_in_history = 5   # Determines how many steps are saved.  Typically choose such that n_iterations is a multiple of this.
    
    # Step 3: Backward optimization starts with random image - define parameters
    start_image_neutral_value = 0
    start_image_noise_level = 0 #0.4 #0.1  #10.1  # 0.1
    start_image_REMOVE_NEGATIVE_VALUES = True
    
    # Do we want to clip optimal input to prescribed range after each iteration of backward optimization?
    CLIP_OPTIMAL_INPUT = False
    zmin = 0 # minimal value in input images
    zmax = 1 # maximal value in input images
    lambda_for_regularization = 0 # regularization parameter
    
    ### Parameter only for plotting
    buffer_in_pixels_for_plotting = 2 # choose a few pixels extra on each side of selected area for plotting
    
    n_layers = len( model.layers )
    layer_names = [layer.name for layer in model.layers]
    print( '# Layers: ' + repr( len(model.layers ) ) )
    
    WANT_BO_PLOTS = True

    for i_layer in range(n_layers):
    #for i_layer in range(1):  # for testing only

        # skip all layers but convolution layers
        if not "conv" in layer_names[i_layer]:
            continue

        # Double check that output of layers has features
        this_layer = model.layers[i_layer]
        this_output_shape = this_layer.output.shape
        if ( len(this_output_shape) < 4 ):  # if output shape does not have 4 members, then there is no feature map to analyze.
            continue

        n_features = this_output_shape[3]
        print('  # of features in this layer: ' + repr(n_features) )

        ########## PLOT results for all features of this layer in a single plot (if desired) and calculate ERF_box #############
        ERF_box = calc_and_plot_backward_optimization_results_one_layer( model, i_layer, ny, nx, nchans, n_iterations, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, lambda_for_regularization, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES, buffer_in_pixels_for_plotting, WANT_BO_PLOTS, backward_optimization_file_start)
        
        # Store values for ERF for this layer (Effective Receptive Field)
        layer_indices_for_ERF.append( i_layer )
        ERF_list.append( ERF_box )

        ########## Plot evolution of optimization for a few features - to get a feeling for convergence of pattern ##############
        for i_feature in range( min(2,n_features) ):    # Just do a few - only consider first two features of each filter
        
            calc_and_plot_backward_optimization_history( model, i_layer, i_feature, ny, nx, nchans, n_iterations, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, lambda_for_regularization, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES, buffer_in_pixels_for_plotting, backward_optimization_file_start)

    ### As side effect of backward optimization we get estimates for the Effective Receptive Field (ERF) of each layer,
    ### i.e. the region of the input image that has pathways to a single pixel in the feature maps of this layer.
    
    
##########################################################################
### Print ERF dimensions for all layers:
print( '\n --- ERF dimensions ---')

if ( len(layer_indices_for_ERF) < 1):
    print('No ERF boundaries calculated.  Use WANT_BACKWARD_OPTIMIZATION_MAPS=True to calculate ERF estimates.')
    
for i in range( len(layer_indices_for_ERF) ):
    i_layer = layer_indices_for_ERF[i]
    this_ERF = ERF_list[i]
    if this_ERF['valid'] == True:
        print( 'Layer {}:   x in [{} {}]  y in [{} {}]'.format(i_layer, this_ERF['xmin'], this_ERF['xmax'], this_ERF['ymin'], this_ERF['ymax']) )
    else:
        print( 'Layer {}:   No valid range.'.format(i_layer) )
        

############### Visualization Task #4 ###############
# Visualize activations of individual neurons:  feature maps

if WANT_FEATURE_MAPS:

    WANT_ONLY_CONV_LAYERS = True

    # Choose whether to analyze test or training sample
    Xdata_for_feature_map = Xdata_test;  sample_description = 'test_sample'   # Use test data
    #Xdata_for_feature_map = Xdata_train; sample_description = 'training_sample'  # Use training data
    
    # Pick a specific sample
    i_sample = 0
    
    # Construct informative caption
    figure_caption_start = NN_string + '  blocks=' + repr(n_encoder_decoder_layers) + '  epochs=' + repr(nepochs) + '   ' + sample_description + ' #' + repr(i_sample)
    
    # Calculate feature maps, generate plots and store in files:
    # One file for each layer.
    calc_and_plot_feature_maps( model, Xdata_for_feature_map, i_sample, WANT_ONLY_CONV_LAYERS, figure_caption_start, feature_map_file_start )
        

############### Visualization Task #5 ###############
# Find max location(s) in each feature map and across feature maps
# This is going to tell us which part of the input sample gives maximal activation.
### Now find maximal/minimal/interesting values in feature maps across all input samples
### and visualize corresponding region of image

if WANT_FEATURE_ACTIVATION_INPUT_PATCHES:

    print( '\nFind patches in input images that maximize feature map.' )

    # Choose whether to analyze test or training sample
    Xdata_for_input_patches = Xdata_test;  sample_description = 'test_sample'   # Use test data
    #Xdata_for_feature_map = Xdata_train; sample_description = 'training_sample'  # Use training data

    # Conv filter can create artifacts at the top/bottom/right/left boundary of activation map
    n_activation_map_boundary_pixels_to_ignore = 3  # number of pixels to_ignore at boundary (at T/B/R/L)

    ####### Build a TF model that can be queried for intermediate layers
    # 1) First extract outputs of all layers
    n_layers = len( model.layers )  # Use all layers
    layer_outputs = [layer.output for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]
    # 2) Create a new model that is a subgraph of the original TF model.
    # This model includes all the information for the intermediate information.
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # For all layers for which we have effective receptive field available
    for counter in range( len(layer_indices_for_ERF) ):   # only use layers for which we calculated ERF
    
        i_layer = layer_indices_for_ERF[counter]
        this_ERF = ERF_list[counter]
            

        ####################################
        ### Plot input patches for specific sample - ACROSS ALL FEATURES
        
        samples_to_analyze = [0,100,200,400] # select samples to analyze
        n_samples =  Xdata_for_input_patches.shape[0]
        
        for i_sample in samples_to_analyze:
            if i_sample < n_samples: # make sure not to exceed number of samples
       
                # each SAMPLE results in a separate image file
                calc_and_plot_MAX_INPUT_PATCHES_one_layer_one_sample_all_features( i_layer, i_sample, activation_model, this_ERF, Xdata_for_input_patches, ny, nx, nchans, sample_description, layer_names, n_activation_map_boundary_pixels_to_ignore, input_patches_file_start )

        ####################################
        ### Plot input patches for specific feature - ACROSS SELECTED SAMPLES
        
        features_to_analyze = [0,1,2,3] # select features to analyze
        n_features = model.layers[i_layer].output.shape[3]

        # select samples to analyze (out-of-range numbers will just be ignored)
        samples_to_analyze = [0,10,20,50,100,200,300,400]
        for i_feature in features_to_analyze:
            if i_feature < n_features:  # make sure not to exceed number of layers
            
                # each FEATURE results in a separate image file
                calc_and_plot_MAX_INPUT_PATCHES_one_layer_one_feature_selected_samples( i_layer, i_feature, activation_model, this_ERF, Xdata_for_input_patches, ny, nx, nchans, sample_description, samples_to_analyze, layer_names, n_activation_map_boundary_pixels_to_ignore, input_patches_file_start )
        
        ####################################


