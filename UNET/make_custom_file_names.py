# MASTER file for setting up different experiments
# - Defines key parameters that might change between experiments.
# - Generates file names with key parameters.

import os
################################################################
def main_string( IS_UNET, n_encoder_decoder_layers, nepochs):
    
    if IS_UNET:
        my_string = 'UNET'  # using Unet
    else:
        my_string = 'SEQ'   # using standard sequential architecture (no skip connections)

    my_string = my_string + '_blocks_' + repr(n_encoder_decoder_layers) + '_epochs_' + repr(nepochs)
    return my_string
################################################################

def ensurefoldersexist(path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok = True)
    
################################################################
def data_file_name( spath, suffix='' ):

    data_file = spath+'/data'+suffix+'.npz'
    
    return data_file
################################################################
def model_file_name( spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs ):

    file_name = spath+'/OUTPUT/MODEL/model_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs) + '.h5'
    ensurefoldersexist(file_name)
    return file_name
################################################################
def history_file_name( spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs ):
     
    file_name = spath+'/OUTPUT/MODEL/history_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs) + '.bin'
    ensurefoldersexist(file_name)
    return file_name
################################################################
def predictions_file_name( spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs ):
     
    file_name = spath+'/OUTPUT/PREDICTIONS/predictions_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs) + '.bin'
    ensurefoldersexist(file_name)
    return file_name
###############################################################
def convergence_plot_file_name( spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs ):
     
    file_name = spath+'/OUTPUT/FIGURES/CONVERGENCE/convergence_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs) + '.png'
    ensurefoldersexist(file_name)
    return file_name
################################################################
def prediction_plot_file_name_start( spath, IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs):

    file_name_start = spath+'/OUTPUT/FIGURES/PREDICTIONS/prediction_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs)
    ensurefoldersexist(file_name_start)
    return file_name_start
################################################################
def feature_map_file_name_start(spath,IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs):

    file_name_start = spath+'/OUTPUT/FIGURES/FEATURE_MAPS/feature_map_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs)
    ensurefoldersexist(file_name_start)
    return file_name_start
################################################################
def heat_map_file_name_start(spath,IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs):

    file_name_start = spath+'/OUTPUT/FIGURES/HEAT_MAPS/heat_map_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs)
    ensurefoldersexist(file_name_start)
    return file_name_start
################################################################
def backward_optimization_file_name_start(spath,IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs):

    file_name_start = spath+'/OUTPUT/FIGURES/BACKWARD_OPTIMIZATION/backward_optimization_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs)
    ensurefoldersexist(file_name_start)
    return file_name_start
################################################################
def input_patches_file_name_start(spath,IS_UNET, my_file_prefix, n_encoder_decoder_layers, nepochs):

    file_name_start = spath+'/OUTPUT/FIGURES/INPUT_PATCHES/input_patches_' + my_file_prefix + '_' + main_string( IS_UNET, n_encoder_decoder_layers, nepochs)
    ensurefoldersexist(file_name_start)
    return file_name_start
################################################################


