#import numpy as np
import pickle
from matplotlib import pyplot as plt

def plot_convergence( historyfile, convergence_plot_file ):

    # open a file, where you stored the pickled data
    my_file = open(historyfile, 'rb')
    history = pickle.load( my_file )
    my_file.close()

    print('\nConvergence - Available functions for plotting:')
    print(history['history'].keys())

    ### Show convergence plots:

    # Print evolution of loss - separately for training and test data
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs[0,0].plot(history['epoch'], history['history']['loss'])
    axs[0,0].set_title('loss (MSE) - training')
    axs[1,0].plot(history['epoch'], history['history']['val_loss'])
    axs[1,0].set_title('loss (MSE) - testing')
    axs[2,0].plot(history['epoch'], history['history']['loss'], 'b--', label="train")
    axs[2,0].plot(history['epoch'], history['history']['val_loss'],'r', label="test")
    axs[2,0].set_title('loss (MSE)')
    axs[2,0].legend(loc='upper right')

    # Print evolution of accuracy - separately for training and test data
    axs[0,1].plot(history['epoch'], history['history']['mean_absolute_error'])
    axs[0,1].set_title('mean abs error (MAE) - training')
    axs[1,1].plot(history['epoch'], history['history']['val_mean_absolute_error'])
    axs[1,1].set_title('mean abs error (MAE) - testing')
    axs[2,1].plot(history['epoch'], history['history']['mean_absolute_error'], 'b--', label="train")
    axs[2,1].plot(history['epoch'], history['history']['val_mean_absolute_error'], 'r', label="test")
    axs[2,1].set_title('mean abs error (MAE)')
    axs[2,1].legend(loc='upper right')


    # Print evolution of accuracy - separately for training and test data
    axs[0,2].plot(history['epoch'], history['history']['my_r_square_metric'])
    axs[0,2].set_title('R square - training')
    axs[1,2].plot(history['epoch'], history['history']['val_my_r_square_metric'])
    axs[1,2].set_title('R square - testing')
    axs[2,2].plot(history['epoch'], history['history']['my_r_square_metric'], 'b--', label="train")
    axs[2,2].plot(history['epoch'], history['history']['val_my_r_square_metric'], 'r', label="test")
    axs[2,2].set_title('R square')
    axs[2,2].legend(loc='upper right')
    
    print('Saving convergnece plot to file ' + repr(convergence_plot_file) )
    plt.savefig( convergence_plot_file )
    
    return



