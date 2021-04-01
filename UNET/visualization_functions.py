import numpy as np
from matplotlib import pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore') # to catch FutureWarnings
    from tensorflow.keras import models
    import tensorflow.keras.backend as K
    import tensorflow as tf

epsilon_for_BO = 0.0001
epsilon_for_activation = 0.0001

########################################################################################
######################           Bounding box arithmetics    ###########################
#    Auxiliary functions for dealing with 2D bounding boxes.
#    These functions simplify handling and manipulating bounding boxes.
########################################################################################

########################################################################################
# Initialize impossible box

def init_infinite_impossible_box():
    # These boxes are impossible, because the mins are larger than the maxs.
    my_box = {'xmin': float("inf"), 'xmax': float("-inf"), 'ymin': float("inf"), 'ymax': float("-inf"), 'region_size' : 0, 'valid': False}
    return( my_box )

########################################################################################
# Initialize valid box from given boundaries.
# Only call this function if you're sure the values are valid!
# To do: Add error handling within this function, i.e. check whether values are valid and return
# message if not.

def init_valid_box( x_min, x_max, y_min, y_max ):
    my_region = (x_max - x_min) * (y_max - y_min)
    my_box = {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max, 'region_size' : my_region, 'valid': True}
    return( my_box )

########################################################################################
# Incorporate boundaries of two boxes into one new box:
#   minimal box that includes all areas of both individual boxes.
#
# Function includes proper error handling if one or both boxes are invalid,
# so no need to check validity ahead of time.

def merge_two_boxes( box1, box2 ):
    
    if box1['valid'] == False:  # if box1 is invalid
        if box2['valid'] == False:  # AND box2 is invalid
            return( init_infinite_impossible_box() )  # return impossible box
        else: # if ONLY box1 is invalid
            return( box2 ) # return box2
    
    # If we get here, we know that box1 is valid
    if box2['valid'] == False:   # if ONLY box2 is invalid
        return( box1 ) # return box1
        
    # Now we know that both boxes are valid:
    # find the smallest box that includes pixels of both boxes.
    x_min = min( box1['xmin'], box2['xmin'] )
    y_min = min( box1['ymin'], box2['ymin'] )
    x_max = max( box1['xmax'], box2['xmax'] )
    y_max = max( box1['ymax'], box2['ymax'] )
    result = init_valid_box( x_min, x_max, y_min, y_max )
    
    return( result )
    
########################################################################################
# Add buffer around box for plotting - but do not exceed image size.

def add_buffer_to_box_and_clip( box, buffer_in_pixels, xmax, ymax ):
    
    # Add buffer, but do not exceed image size
    x_min_new = max( box['xmin'] - buffer_in_pixels, 0 )
    y_min_new = max( box['ymin'] - buffer_in_pixels, 0 )
    x_max_new = min( box['xmax'] + buffer_in_pixels, xmax )
    y_max_new = min( box['ymax'] + buffer_in_pixels, ymax )
    
    result = init_valid_box( x_min_new, x_max_new, y_min_new, y_max_new )
    return( result )
    
########################################################################################
# This function determines the minimal horizontal rectangle in an image containing all pixels with non-zero values.
# Sample use: Combined with backward optimization, can use this to find effective receptive field (ERF) of a feature.

def find_minimal_bounding_box_in_2D_image ( this_2D_image ):
    
    # find indices of all non-zero values
    select_indices = np.where( abs(this_2D_image) > 0 )
    y_range = select_indices[0]  # all y-coordinates
    x_range = select_indices[1]  # all x-coordinates
    
    # Default is invalid box:
    my_box = init_infinite_impossible_box()
    
    # If there are any non-zero pixels: update box
    if len(x_range) > 0 :  # if there is at least one non-zero pixel
        x_min = min( x_range )
        x_max = max( x_range )
        y_min = min( y_range )
        y_max = max( y_range )
        # Create valid box
        my_box = init_valid_box( x_min, x_max, y_min, y_max )
        
    return( my_box )

########################################################################################
# This function does the same as find_minimal_bounding_box_in_2D_image,
# but combines results across multiple channels.

def find_minimal_bounding_box_in_multi_channel_image ( this_MC_image ):

    n_chans = this_MC_image.shape[-1]  # last dimension is # of channels
    
    # Initialize box to keep track of overall min and max coordinates
    my_box = init_infinite_impossible_box()
    
    for i_chan in range( n_chans ):
        this_image = this_MC_image[:,:,i_chan]
        my_new_box = find_minimal_bounding_box_in_2D_image( this_image )
        
        if my_new_box['valid'] == False:   # no interesting pixels found
            continue
        
        my_box = merge_two_boxes( my_box, my_new_box )
                      
    # Done with all channels.  Communicate results.
    """
    if ( my_box['valid'] ):
        print('   Boundaries across all channels:  x=[' + repr(my_box['xmin']) + ',' + repr(my_box['xmax']) + ']   y=[' + repr(my_box['ymin']) + ',' + repr(my_box['ymax']) +']')
    
    else:
        print('   Boundaries across all channels:  INVALID - no pixels of interest found')
    """
    return( my_box )

########################################################################################
######################       Other auxiliary functions       ###########################
########################################################################################


########################################################################################
def find_symmetric_range_for_colormap( my_2D_image, my_epsilon ):
    # Determine max and min, then create color scale that uses symmetric range around 0, i.e. of the form [-M,M].
    z_min = np.amin(my_2D_image)  # min value in image
    z_max = np.amax(my_2D_image)  # max value in image
    z_abs_max = max( abs(z_min), abs(z_max) )
    
    non_zero_values_exist = True
    # Deal with degeneracy, i.e. if z_min=z_max, add a small number to avoid non-zero range for colormap.
    if z_abs_max < my_epsilon:
        z_abs_max = max( z_abs_max, my_epsilon )
        non_zero_values_exist = False

    return( z_abs_max, z_min, z_max, non_zero_values_exist )

########################################################################################
# This function constructs a random input of prescribed size.
# Sample use: as starting point for backward optimization.

def create_random_input_sample( input_ny, input_nx, input_nchans, neutral_value, noise_level, REMOVE_NEGATIVE_VALUES ):
    
    # Generate random image
    input_sample = np.random.random( (1, input_ny, input_nx, input_nchans) ) * noise_level + neutral_value
    
    if REMOVE_NEGATIVE_VALUES:
        input_sample = np.maximum( input_sample, 0 ) # Remove negative values
        
    return( input_sample )

########################################################################################

########################################################################################
######################  Functions for plotting predictions  ###########################
########################################################################################



########################################################################################
######################  Functions for feature maps  ###########################
########################################################################################

def calc_and_plot_feature_maps( model, Xdata, i_sample, WANT_ONLY_CONV_LAYERS, figure_caption_start, feature_map_file_start ):

    # This code follows the approach outlined in Cholet's book
    # "Deep Learning with Python" (Section 5.4.1)

    print( '\nVisualize activations of intermediate neurons (feature maps)' )
    
    # Extract dimensions
    my_shape = Xdata.shape
    ny = my_shape[1]
    nx = my_shape[2]
    nchans = my_shape[3]

    img_tensor = Xdata[i_sample,:,:,:].reshape(1,ny,nx,nchans)
    # Sample ready

    n_layers = len( model.layers )  # Use all layers

    ####### Build a new TF model that has all layers we care about and that can be queried for intermediate layers

    # 1) First extract outputs of all layers
    layer_outputs = [layer.output for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]

    # 2) Create a new model that is a subgraph of the original model by tensor flow.
    # This model contains only the first n_layers, but includes the intermediate information.
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # 3) Feed sample into this model and get the activations of its intermediate layers
    activations = activation_model.predict( img_tensor )

    for i_layer in range(n_layers):

        # If desired: skip all layers but convolution layers
        if (WANT_ONLY_CONV_LAYERS==True and not "conv" in layer_names[i_layer]):
            continue
            
        # Extract activations for one layer
        my_layer_activation = activations[i_layer]
        print( my_layer_activation.shape )  # Shape should be (1,nx,ny,n_filters)

        if len(my_layer_activation.shape) < 4:
            print('Layer ' + repr(i_layer) + ' does not have proper activation map.')
            continue

        # 5) Plot all feature maps for one layer
        # How many features are there?
        n_feature_maps = my_layer_activation.shape[-1]  # last element provides # filters in that layer

        if n_feature_maps < 1:
            continue

        image_width = my_layer_activation[0,:,:,0].shape[1]
        print( 'Layer ' + repr(i_layer) + ' (' + layer_names[i_layer] + '): image width is ' + repr(image_width) + ' pixels')

        n_figures_per_row = 4
        n_cols = n_figures_per_row
        n_rows = n_feature_maps // n_figures_per_row
        
        # deal with case when n_feature_maps is not divisible by n_figures_per_row --> left-over plots
        # --> need one more row
        if ( n_rows * n_figures_per_row < n_feature_maps ):
            n_rows = n_rows+1

        #print( '   ' + repr(n_feature_maps) + ' feature maps: ' + repr(n_rows) + ' rows, ' + repr(n_cols) + ' cols')

        f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        axes = axes.reshape((-1,n_cols)) # deal with the case of n_rows=1.  Otherwise subplots throws an error.

        for i_feature in range(n_feature_maps):

            i_row = i_feature // n_figures_per_row
            i_col = i_feature - i_row * n_figures_per_row
            
            my_image = my_layer_activation[0,:,:,i_feature].reshape(image_width,image_width)
            # Create color scale that uses symmetric range around 0, i.e. of the form [-M,M].
            z_abs_max, z_min, z_max, non_zero_values_exist = find_symmetric_range_for_colormap( my_image, epsilon_for_activation )
            # z_abs_max = max( 1.0, z_abs_max )  # Use [-1,1] as range, unless z_abs_max > 1.
            # In RdGy:  gray is positive, white is 0, red is negative
            axes[i_row,i_col].imshow( my_image, cmap='RdGy', clim=(-z_abs_max,z_abs_max) )
            axes[i_row,i_col].set_title('Feature {:#2}   [{:#.1F} , {:#.1F}]'.format(i_feature,z_min,z_max) )
            if not non_zero_values_exist:
                # draw a red box with title "No activation" in middle of box
                axes[i_row,i_col].text(image_width//2, image_width//2, 'No activation', style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 1})
            
        # Finish up: add caption and save entire image to file
        f.suptitle( figure_caption_start + '  -  Layer ' + repr(i_layer) + ':  ' + layer_names[i_layer], fontsize=24 )
        
        my_plot_filename = feature_map_file_start + '_sample_' + repr(i_sample) + '_activation_l_' + repr(i_layer) + '_' + layer_names[i_layer] +'.png'

        print('Saving sample activation plot to file ' + repr(my_plot_filename) )
        plt.savefig( my_plot_filename )
        plt.close()
        
    return



########################################################################################
######################  Functions for plotting max input patches  ###########################
########################################################################################


############# Plot input patches for single i_layer, single i_feature, and ALL samples ##################
def calc_and_plot_MAX_INPUT_PATCHES_one_layer_one_feature_all_samples( i_layer, i_feature, activation_model, this_ERF, Xdata_for_input_patches, ny, nx, nchans, sample_description, layer_names, n_activation_map_boundary_pixels_to_ignore, input_patches_file_start ):

    n_samples = Xdata_for_input_patches.shape[0]
    
    ######## Create big plot - each row is for one sample
    n_figures_per_row = 5  # Each row: feature map itself, plus patches of 4 input channel
    n_cols = n_figures_per_row
    n_rows = n_samples
    
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.reshape((-1,n_cols)) # deal with the case of n_rows=1.  Otherwise subplots throws an error.
    
    for i_sample in range( n_samples ):
    
        i_row = i_sample  # each row represents one sample

        # Operations specific to sample:
        # Feed sample into this shortened model and get the activations of its intermediate layers
        img_tensor = Xdata_for_input_patches[i_sample,:,:,:].reshape(1,ny,nx,nchans)  # Sample ready
        activations = activation_model.predict( img_tensor )
        
        # Extract activations for this layer and this sample
        my_layer_activation = activations[i_layer]
        #print( my_layer_activation.shape )  # Shape should be (1,nx,ny,n_filters)

        if len(my_layer_activation.shape) < 4:
            print('WARNING:  Layer ' + repr(i_layer) + ' does not have proper activation map.')
            #continue

        # Consider all feature maps for one layer
        n_feature_maps = my_layer_activation.shape[-1]  # last element provides # filters in that layer
        if n_feature_maps < 1:
            print(' Warning in MAX_INPUT_PATCHES: layer ' + repr(i_feature) + ' does not have any features' )

        # Generate row of images - for one feature and one sample
        #print('Input patches for:  Layer ' + repr(i_layer) + ' feature ' + repr(i_feature) + '  ' + sample_description + ' #' + repr(i_sample) )

        calc_and_plot_one_row_MAX_INPUT_PATCHES_one_layer_one_feature_one_sample( axes, i_row, i_layer, i_feature, i_sample, my_layer_activation, this_ERF, Xdata_for_input_patches, ny, nx, nchans,  n_activation_map_boundary_pixels_to_ignore  )
        
    # Add label for each row
    rows = ['Sample {}  '.format(row) for row in range(n_feature_maps)]
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    #f.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    f.suptitle( 'Layer ' + repr(i_layer) + ':  ' + layer_names[i_layer] + '   Feature ' + repr(i_feature) + '   All samples', fontsize=24 )

    my_plot_filename = input_patches_file_start + '_layer_' + repr(i_layer) + '_feature_' + repr(i_feature) + '.png'

    print('Saving high activation input patch to file ' + repr(my_plot_filename) )
    plt.savefig( my_plot_filename )
    plt.close()
    return


############# Plot input patches for single i_layer, single i_sample, and ALL features ##################

def calc_and_plot_MAX_INPUT_PATCHES_one_layer_one_sample_all_features( i_layer, i_sample, activation_model, this_ERF, Xdata_for_input_patches, ny, nx, nchans, sample_description, layer_names, n_activation_map_boundary_pixels_to_ignore, input_patches_file_start ):

    # Operations specific to sample:
    # Feed sample into this shortened model and get the activations of its intermediate layers
    img_tensor = Xdata_for_input_patches[i_sample,:,:,:].reshape(1,ny,nx,nchans)  # Sample ready
    activations = activation_model.predict( img_tensor )
    
    # Extract activations for this layer and this sample
    my_layer_activation = activations[i_layer]
    #print( my_layer_activation.shape )  # Shape should be (1,nx,ny,n_filters)

    if len(my_layer_activation.shape) < 4:
        print('WARNING:  Layer ' + repr(i_layer) + ' does not have proper activation map.')
        #continue

    # Consider all feature maps for one layer
    n_feature_maps = my_layer_activation.shape[-1]  # last element provides # filters in that layer
    if n_feature_maps < 1:
        print(' Warning in MAX_INPUT_PATCHES: layer ' + repr(i_feature) + ' does not have any features' )

    ######## Create big plot - each row is for one feature
    n_figures_per_row = 5  # Each row: feature map itself, plus patches of 4 input channel
    n_cols = n_figures_per_row
    n_rows = n_feature_maps

    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.reshape((-1,n_cols)) # deal with the case of n_rows=1.  Otherwise subplots throws an error.

    #print( '   ' + repr(n_feature_maps) + ' features maps: ' + repr(n_rows) + ' rows, ' + repr(n_cols) + ' cols')

    for i_feature in range(n_feature_maps):

        i_row = i_feature

        ######################## START OF FUNCTION (i_layer,i_feature,i_sample) ##################
        # Generate row of images - for one feature and one sample
        #print('Input patches for:  Layer ' + repr(i_layer) + ' feature ' + repr(i_feature) + '  ' + sample_description + ' #' + repr(i_sample) )

        calc_and_plot_one_row_MAX_INPUT_PATCHES_one_layer_one_feature_one_sample( axes, i_row, i_layer, i_feature, i_sample, my_layer_activation, this_ERF, Xdata_for_input_patches, ny, nx, nchans,  n_activation_map_boundary_pixels_to_ignore  )
    
    # Add label for each row
    rows = ['Feature {}  '.format(row) for row in range(n_feature_maps)]
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    #f.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    f.suptitle( 'Sample ' + repr(i_sample) + '  Layer ' + repr(i_layer) + ':  ' + layer_names[i_layer] + '    All features', fontsize=24 )

    my_plot_filename = input_patches_file_start + '_layer_' + repr(i_layer) + '_sample_' + repr(i_sample) + '.png'

    print('Saving high activation input patch to file ' + repr(my_plot_filename) )
    plt.savefig( my_plot_filename )
    plt.close()
    return

######################## input patches for single layer, single feature, and single sample) ##################
### This function is used by the composite functions above ###

def calc_and_plot_one_row_MAX_INPUT_PATCHES_one_layer_one_feature_one_sample( axes, i_row, i_layer, i_feature, i_sample, my_layer_activation, this_ERF, Xdata_for_input_patches, ny, nx, nchans,  n_activation_map_boundary_pixels_to_ignore  ):

    print('**KH** my_layer_activation shape=',my_layer_activation.shape)
    image_width = my_layer_activation[0,:,:,0].shape[1]  # assuming a square image
    print('**KH** image_width=',image_width)

    # Retrieve actiation map and reshape to 2D image
    my_matrix = my_layer_activation[0, :, :, i_feature].reshape(image_width, image_width)
    print('**KH** ny,nx=',ny,nx)
    print('**KH** my_matrix shape=',my_matrix.shape)

    # Cut out smaller area - to deal with artifacts at boundary
    if ( image_width > 2 * n_activation_map_boundary_pixels_to_ignore ):   # check whether there is space to cut
        n_ignore = n_activation_map_boundary_pixels_to_ignore
    else:
        n_ignore = 0  # can't cut off boundary, because there would be nothing left

    # interior activation map
    activation_wo_boundary = my_matrix[ n_ignore : image_width-1-n_ignore, n_ignore : image_width-1-n_ignore]
    # Find location of maximal activation
    index_pair = np.unravel_index(np.argmax(activation_wo_boundary, axis=None), activation_wo_boundary.shape)

    # Convert to coordinates of full activation map (with boundaries):  just add n_ignore to x and y
    index_pair = (index_pair[0] + n_ignore, index_pair[1] + n_ignore)
    i_y = index_pair[0]  # y comes first
    i_x = index_pair[1]  # x comes next
    feature_value = my_matrix[index_pair]
    #print( 'Max location: (' + repr(i_x) + ',' + repr(i_y) + ').     Feature map value: ' + repr( feature_value ) )

    # Calculate corresponding patch in input layer = patch that results in this feature map value.
    # Method: Take ERF estimated for center point of this layer, and translate according to the index pair.

    # number of pixels in this layer
    ny_this_layer = my_layer_activation[0,:,:,0].shape[0]
    nx_this_layer = my_layer_activation[0,:,:,0].shape[1]

    # The following assumes that
    #  a) the feature map never has more pixels than the input and
    #  b) the ratio is a whole number
    # Both are satisfied for standard encoder-decoder networks and Unets.
    y_factor = ny // ny_this_layer  # how many pixels in input layer for each pixel in this layer.
    x_factor = nx // nx_this_layer

    # coordinates used for calculating ERF
    i_y_center = ny_this_layer // 2
    i_x_center = nx_this_layer // 2

    # Shift boundaries accordingly, but don't exceed input image dimensions
    my_y_min = max( 0, this_ERF['ymin'] + y_factor * ( i_y - i_y_center ) )
    my_y_max = min( ny-1, this_ERF['ymax'] + y_factor * ( i_y - i_y_center ) )
    my_x_min = max( 0, this_ERF['xmin'] + x_factor * ( i_x - i_x_center ) )
    my_x_max = min( ny-1, this_ERF['xmax'] + x_factor * ( i_x - i_x_center ) )


    ######### GENERATING ONE ROW OF IMAGES ##########
    ### Now add the feature map
    i_col = 0
    my_image = my_matrix  # feature activation map
    print('**KH** my_image shape = ',my_image.shape)
    z_abs_max, z_min, z_max, non_zero_values_exist = find_symmetric_range_for_colormap( my_image, epsilon_for_activation )
    #z_abs_max = max( 0.5, z_abs_max )  # Use [-1,1] as range, unless z_abs_max > 1.
    
    # RdGy: Gray is positive, white is 0, Red is negative.
    axes[i_row,i_col].imshow( my_image, cmap='RdGy', clim=(-z_abs_max,z_abs_max) )
    axes[i_row,i_col].set_title('Feature map [{:#.2F},{:#.2F}] \n max value: f({},{})={:#.2f}'.format(z_min,z_max,i_x,i_y,feature_value) )

    if ( feature_value > epsilon_for_activation ):  # if there's actually a non-zero-value
        # Add circle in feature map at hot spot location = location for input patches
        my_circle = plt.Circle((i_x, i_y), 6, color='red', fill=False)
        axes[i_row,i_col].add_artist( my_circle )
    else:
        # draw a red box with title "No activation" in middle of box
        axes[i_row,i_col].text(i_y_center, i_x_center, 'No activation', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 1})
                    
    channel_text = ['ABI #1', 'ABI #2', 'ABI #3', 'GLM']
    for i_chan in range( nchans ):
        #i_row = i_feature
        i_col = i_chan + 1
            
        if ( feature_value > epsilon_for_activation ):  # only draw input patch if there was actually activation
            this_image = Xdata_for_input_patches[i_sample,:,:,i_chan].reshape(ny,nx)
            ### Cut out the patch
            this_patch = this_image[ my_y_min:my_y_max, my_x_min:my_x_max]
            z_abs_max, z_min, z_max, non_zero_values_exist = find_symmetric_range_for_colormap( this_patch, epsilon_for_activation )
            #z_abs_max = max( 0.5, z_abs_max )  # Use [-0.5,0.5] as range, unless z_abs_max > 0.5.
            axes[i_row,i_col].imshow(this_patch, cmap='seismic', clim=(-z_abs_max,z_abs_max) )
            #axes[i_row,i_col].set_title('f(' + repr(i_x) + ',' + repr(i_y) + ') = ' + repr(feature_value) )
            axes[i_row,i_col].set_title( '{}  [{:#.2F},{:#.2F}]'.format(channel_text[i_chan], z_min, z_max) )
        else:
            axes[i_row,i_col].set_visible(False)
            
    return
########################################################################################


########################################################################################
######################  Functions for backward optimization  ###########################
########################################################################################

########################################################################################
# Plot end results for all features in a layer in a single plot

def calc_and_plot_backward_optimization_results_one_layer( model, i_layer, ny, nx, nchans, n_iterations, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, my_lambda, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES, buffer_in_pixels, WANT_PLOTS, backward_optimization_file_start):

    layer_names = [layer.name for layer in model.layers]
    
    # Calculate Effective Receptive Field (ERF) for this layer
    # Initiate with impossible values
    ERF_box = init_infinite_impossible_box()
    
    this_layer = model.layers[i_layer]
    
    n_features = this_layer.output.shape[3]
    
    # Create one big plot for all features of this channel
    n_rows = n_features  # Each row = one feature map
    n_cols = nchans # Each column represents one channel.
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.reshape((-1, n_cols))  # Deal with case when n_rows=1.  Otherwise subplots throws an error.
        
    for i_feature in range( n_features ):

        #print('Perform Backward Optimization - for layer #' + repr(i_layer) + ': ' + layer_names[i_layer] + '    Feature ' + repr(i_feature) )
        
        ########### Perform backward optimization ###
        BO_start_input, BO_final_result, BO_result_history, BO_index_history, BO_first_gradient = backward_optimization( model, layer_names[i_layer], i_feature, ny, nx, nchans, n_iterations, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, my_lambda, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES)
        
        # Add plot as subplot
        # Plot preparation: find relevant area in input image - this is an estimate of ERF.
        ## Find min/max x-/y-coordinate for which optimal input deviates from start input
        image_for_boundaries = (BO_final_result - BO_start_input).reshape(ny,nx,nchans)
        my_new_box = find_minimal_bounding_box_in_multi_channel_image( image_for_boundaries )

        if my_new_box['valid']:
        
            # Update ERF boundary:
            # Keep the smallest x/y_min and the largest x/y_max.
            ERF_box = merge_two_boxes( ERF_box, my_new_box )
            
            if WANT_PLOTS:
                # Add a few pixels on each side just for plotting
                # Yes, here we need first nx, then ny.
                my_buffered_box = add_buffer_to_box_and_clip( my_new_box, buffer_in_pixels, nx-1, ny-1 )

                this_result = BO_final_result
                
                for i_chan in range(nchans):
                    i_row = i_feature
                    i_col = i_chan
                    # extract image for single channel
                    my_image = this_result[0,:,:,i_chan].reshape(ny,nx)
                    # Create color scale that uses symmetric range around 0, i.e. of the form [-M,M].
                    z_abs_max, z_min, z_max, non_zero_values_exist = find_symmetric_range_for_colormap( my_image, epsilon_for_BO )
                    # add image to overall plot using that color scale
                    # In seismic:  red is positive, white is 0, blue is negative
                    axes[i_row,i_col].imshow( my_image, cmap='seismic', clim=(-z_abs_max,z_abs_max) )
                    axes[i_row,i_col].set_title('\nFeature {:#2}   [{:#.1F} , {:#.1F}]'.format(i_feature,z_min,z_max) )
                    axes[i_row,i_col].set_xlim(my_buffered_box['xmin'],my_buffered_box['xmax'])
                    axes[i_row,i_col].set_ylim(my_buffered_box['ymin'],my_buffered_box['ymax'])
                    
        else:
            for i_chan in range(nchans):
                i_row = i_feature
                i_col = i_chan
                axes[i_row,i_col].set_visible(False)

    if WANT_PLOTS:
        # Save plot
        my_plot_filename = backward_optimization_file_start + '_layer_' + repr(i_layer) + '.png'
        plt.savefig( my_plot_filename )
        plt.close()
        print('Saved plot to file ' + my_plot_filename  )
        
    return( ERF_box )


########################################################################################

def calc_and_plot_backward_optimization_history( model, i_layer, i_feature, ny, nx, nchans, n_iterations, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, my_lambda, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES, buffer_in_pixels, backward_optimization_file_start):

    layer_names = [layer.name for layer in model.layers]

    #print('Perform Backward Optimization - for layer #' + repr(i_layer) + ': ' + layer_names[i_layer] + '    Feature ' + repr(i_feature) )
    
    ########### Perform backward optimization ###
    BO_start_input, BO_final_result, BO_result_history, BO_index_history, BO_first_gradient = backward_optimization( model, layer_names[i_layer], i_feature, ny, nx, nchans, n_iterations, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, my_lambda, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES)

    ########### PLOT RESULTS #############
    ### Plotting entire history
    my_plot_filename = backward_optimization_file_start + '_layer_' + repr(i_layer) + '_feature_' + repr(i_feature) + '_evolution.png'
    plot_backward_optimization_history( BO_start_input, BO_final_result, BO_result_history, BO_index_history, ny, nx, nchans, i_layer, i_feature, buffer_in_pixels, my_plot_filename )
    
    return

########################################################################################
# Perform Backward Optimization and plot history of optimized inputs to show otpimiztion process step by step.

def plot_backward_optimization_history( BO_start_input, BO_final_result, BO_result_history, BO_index_history, ny, nx, nchans, i_layer, i_feature, buffer_in_pixels, my_plot_filename ):

    # Plot preparation: find relevant area in input image - this is an estimate of ERF.
    ## Find min/max x-/y-coordinate for which optimal input deviates from start input
    image_for_boundaries = (BO_final_result - BO_start_input).reshape(ny,nx,nchans)
    my_box = find_minimal_bounding_box_in_multi_channel_image ( image_for_boundaries )

    # If no pixels deviate from neutral --> all pixel values are identical --> Nothing to plot.
    if not my_box['valid']:
        return
        
    # Add small a few pixels on each side just for plotting
    # buffer_in_pixels = 2 # number of pixels to add as buffer on each side, but not exceeding image size
    
    my_buffered_box = add_buffer_to_box_and_clip( my_box, buffer_in_pixels, nx, ny )
    
    n_history_steps_stored = len(BO_index_history)

    # Create one big plot with all the iterations
    n_rows = n_history_steps_stored  # Each row = one iteration
    n_cols = nchans # Each column represents one channel.
    f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.reshape((-1, n_cols))  # Deal with case when n_rows=1.  Otherwise subplots throws an error.

    # Go through all iterations of optimization
    for i_history in range( n_history_steps_stored ):

       # Extract channel images for this iteration from history
       this_result = BO_result_history[i_history]

       # Add the images for all channels for this iteration in one row
       for i_chan in range( nchans ):
           i_row = i_history
           i_col = i_chan
           # extract image for single channel
           my_image = this_result[0,:,:,i_chan].reshape(ny,nx)

           # Create color scale that uses symmetric range around 0, i.e. of the form [-M,M].
           z_abs_max, z_min, z_max, non_zero_values_exist = find_symmetric_range_for_colormap( my_image, epsilon_for_BO )
           # In seismic:  red is positive, white is 0, blue is negative
           axes[i_row,i_col].imshow( my_image, cmap='seismic', clim=(-z_abs_max,z_abs_max) )
           axes[i_row,i_col].set_title('\nStep {:#2}   [{:#.1F} , {:#.1F}]'.format(BO_index_history[i_history],z_min,z_max) )
           axes[i_row,i_col].set_xlim(my_buffered_box['xmin'],my_buffered_box['xmax'])
           axes[i_row,i_col].set_ylim(my_buffered_box['ymin'],my_buffered_box['ymax'])
           if not non_zero_values_exist:
                # draw a red box with title "Zero signal" in middle of box
                axes[i_row,i_col].text(ny//2, nx//2, 'Zero signal', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 1})

    plt.savefig( my_plot_filename )
    plt.close()
    print('Saved plot to file ' + my_plot_filename  )
    return

########################################################################################
def backward_optimization( model, layer_name, feature_index, input_ny, input_nx, input_nchans, n_steps, grad_step_size, save_every_n_th_figure_in_history, i_loss_function, CLIP_OPTIMAL_INPUT, zmin, zmax, my_lambda, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES ):

    #print('\n')
    #print(layer_name)
    #print('Feature ' + repr(feature_index) + '\n')
    
    #print('n_steps = ' + repr(n_steps) )
    #print('save_every_n_th_figure_in_history = ' + repr(save_every_n_th_figure_in_history) )

    # define which layer we want and extract it from model
    spec_layer_output = model.get_layer(layer_name).output
    
    # Get dimenstion to find a central point of feature map below
    feature_map_n_x = spec_layer_output.shape[2]
    feature_map_n_y = spec_layer_output.shape[1]

    # define new model that goes from input to desired output layer
    activation_model = models.Model(inputs=model.input, outputs=spec_layer_output)
    #activation_model.summary()  # for debugging

    # Keep track of evolution of input change
    input_history = []  # start with empty list
    index_history = []  # also store the corresponding indices (steps)

    start_input = create_random_input_sample( input_ny, input_nx, input_nchans, start_image_neutral_value, start_image_noise_level, start_image_REMOVE_NEGATIVE_VALUES)
    
    # Add original image as first element in history list
    input_history.append( start_input )
    index_history.append( 0 )  # This is step 0

    # Convert np-array to tf variable.
    # Need to specify float32, because np uses float64 and tf requires explicit type conversion.
    my_input = tf.Variable( start_input, dtype=tf.float32 )
        
    for i_step in range(n_steps):
    
        #print('Iteration #' + repr(i_step) )
        #print( 'Step ' + repr(i_step) + ':')

        ############## START OF GRADIENT CALCULATION ###################
        with tf.GradientTape() as tape:
            my_activation = activation_model( my_input )
            
            if i_loss_function == 0:
                # Usual loss function:  maximize mean activation level in feature map
                # --> maximize activation throughout all locations
                my_loss = K.mean( my_activation[:,:,:,feature_index] )
            
            if i_loss_function == 1:
                # Alternative: maximize mean activation at a single location
                # Use center coordinates of feature_map as location
                my_loss = my_activation[:, feature_map_n_y//2, feature_map_n_x//2, feature_index]
            
            # add regularization:
            # penalize pixels that are OUTSIDE POSSIBLE VALUE RANGE of input

            # note the - (minus sign) in front of my_lambda, because we are using steepest ASCENT,
            # thus penalties here must REDUCE the value of the loss function (not INCREASE as usual).
        
            #print('Activation loss:' + repr(my_loss) )
            
            my_loss = ( my_loss
            - my_lambda * tf.reduce_sum(  # penalize all pixel values below zmin
                tf.where( tf.less( my_input, zmin ),   # if my_input < zmin
                tf.subtract( zmin, my_input ),  # if true: penalty = (zmin - my_input) = amount of violation (positive)
                tf.zeros_like(my_input)  # if not true: no penalty, i.e. zeros but in proper tensor form
                ))
            - my_lambda * tf.reduce_sum(  # penalize all pixel values above zmax
                tf.where( tf.less( zmax, my_input ),   # if my_input > zmax
                tf.subtract( my_input, zmax ),  # if true: penalty = (my_input - zmax) = amount of violation (positive)
                tf.zeros_like(my_input)  # if not true: no penalty, i.e. zeros but in proper tensor form
                )) )
            
            #print('Total loss:' + repr(my_loss) )

        my_gradients = tape.gradient( my_loss, my_input )
        ############## END OF GRADIENT CALCULATION ###################
        
        if i_step == 0:
            first_gradient = my_gradients   # save the first gradient to return BEFORE normalization
            
        #print( '  Gradient values before normalization: [' + repr(np.amin(my_gradients)) + ',' + repr(np.amax(my_gradients)) + ']')

        # Normalize gradient  # this was blowing up gradient to have max/min value of 100s!
        #my_gradients /= ( K.sqrt( K.mean( K.square( my_gradients ) ) ) + 1e-5 )

        #print( K.sqrt( K.mean( K.square( my_gradients ) ) ) )
        
        #print( '  Gradient values after normalization: [' + repr(np.amin(my_gradients)) + ',' + repr(np.amax(my_gradients)) + ']')
        
        # Cut off negative gradients - if desired - this is replaced by clipping results below
        #if gradients_REMOVE_NEGATIVE_VALUES:
        #    my_gradients = tf.maximum( my_gradients, 0 )
            
        #print( '  Input values before modification: [' + repr(np.amin(my_input)) + ',' + repr(np.amax(my_input)) + ']')

        # Add scaled gradient to current version
        my_input = tf.Variable( tf.add( my_input, my_gradients * grad_step_size ) )
        
        #print( '  Input values after modification: [' + repr(np.amin(my_input)) + ',' + repr(np.amax(my_input)) + ']')
        
        # Clip values that are below minimum or above maximum allowed for input images
        if ( CLIP_OPTIMAL_INPUT ):
            my_input = tf.Variable( tf.maximum( my_input, zmin ) )
            my_input = tf.Variable( tf.minimum( my_input, zmax ) )
            
        #print( '  Input values after modification: [' + repr(np.amin(my_input)) + ',' + repr(np.amax(my_input)) + ']')
        
        # Save selected plots to history
        if ( (i_step + 1) % save_every_n_th_figure_in_history == 0 ):
            # convert current image to numpy array and add to history
            input_history.append( my_input.numpy() )
            index_history.append( i_step + 1 )  # start counting iterations steps at 1 (not 0)

        final_input = my_input.numpy()  # saving last image separately

    return( start_input, final_input, input_history, index_history, first_gradient )
########################################################################################




############## Old stuff below - ignore #################

"""
    # Backwards optimization with Keras
    # This is the version from Cholet's book "Deep Learning with Python",
    # Section 5.4.2.
    
    # But it does not work in TF 2.0...
    
    # Usual loss function:  maximize mean activation level in feature map
    # --> maximize actication throughout all locations
    mean_loss = K.mean( layer_output[:,:,:,feature_index] )

    # Alternative:  maximize mean activatiomn at a single location
    # Use center coordinates of feature_map
    center_loss = layer_output[:, fm_nx//2, fm_ny//2, feature_index]
    
    my_loss = center_loss
    
    # Step 3: Define which gradient we want - which loss function, and w.r.t. to input image
    # Keras version - does not work with TF 2.0 default mode
    my_grads = K.gradients( my_loss, model.input )[0]
    # K.gradients returns a list of tensors, in this case one.  [0] refers to this first one.

    # Step 4: Specify normalization for gradient
    # Divide by L2 norm (sqrt of average of squares in tensor)
    # - then add small nunmber in denominator to avoid division by 0
    grads /= ( K.sqrt( K.mean( K.square( my_gradients ) ) ) + 1e-5 )
    
    # Step 5: Generate FUNCTION that maps input to loss and gradient
    
    # Define a function that extracts the relevant parts of the model's compute graph.
    # Keras' functon command allows you to do that using the funnction command as follows:
    # Specify a LIST of desired inputs, to be mapped to a LIST of desired outputs,
    # and Keras extracts the corresponding mapping function for us from the compute graph.
    get_loss_grad = K.function( [model.input], [my_loss,my_grads] )
    
    # This function can now be called with an input sample to calculate the loss and grad.

    
    # Now modify input sample so that it maximizes chosen activation
    stepsize = 1.0
    n_steps = 40
    for i in range( n_steps ):
       loss_value, grads_value = get_loss_grad( [input_image_data] )
       # Note that we are ADDING the gradient, so this is gradient ASCENT,
       # to MAXIMIZE the loss function.
       input_image_data += grads_value * step_size
    
    modified_image = input_image_data[0]
    
    # Test dimensions:
    print( 'Modified input image should have ' + repr(input_nchans) + ' channels.')
    print( modified_image.shape )
    
    return( modified_image )
"""
########################################################################################
