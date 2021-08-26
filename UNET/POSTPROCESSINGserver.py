#PHD_proj_POSTPROCESS
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:00:25 2020
#edited 6 JAN 2021
#edited TUE Apr 26 PPS
@author: cpasilla
"""

##### post process applications


####using the model make the predictands
#from netCDF4 import Dataset
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from custom_model_elements import *
import matplotlib.pyplot as plt
import math
import normalize
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats, odr


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12.0/2, 8.0/2)

spath = "/zdata2/cpasilla/MARCH2019_ALL/ML_INPUT/DNB_full_moon_norm-12_predictors/"
#spath = "/Users/cpasilla/PHD_WORK/TESTING/"
#%%
#load the data and model
data = np.load(spath + 'data.npz') #"data_file"
model = load_model(spath + 'OUTPUT/MODEL/model_C1_UNET_blocks_3_epochs_50.h5', custom_objects = { \
           "my_r_square_metric": my_r_square_metric,\
           "my_mean_squared_error_weighted_genexp": my_mean_squared_error_weighted_genexp,\
           "loss": my_mean_squared_error_weighted_genexp(),\
           "my_csi20_metric": my_csi20_metric,\
           "my_csi35_metric": my_csi35_metric,\
           "my_csi50_metric": my_csi50_metric,\
           "my_bias20_metric": my_bias20_metric,\
           "my_bias35_metric": my_bias35_metric,\
          "my_bias50_metric": my_bias50_metric}) 
                   # "model_file"


#whichtruth = data['y_channel'][0]
#TODO - with new array adjust this 
whichtruth = 'DNB_full_moon_norm'
truth = data['Ydata_train']
prediction= model.predict(data['Xdata_train'])

#%%
#renormalize data ( raw and ML outputs)

xx = normalize.denormalize(whichtruth, truth)
yy = normalize.denormalize(whichtruth, prediction)

print( 'DNB raw truth max is', np.nanmax(yy))
print( 'DNB raw truth min is', np.nanmin (yy))
print( 'ML raw rad max is', np.max(xx))
print( 'ML raw rad min is', np.min(xx))


##
#calculate RMSE
from sklearn.metrics import mean_squared_error 

fxx=xx.flatten()
fyy= yy.flatten()
RMSDtruth =np.sqrt(mean_squared_error(fxx,fyy))
print("the real RMSE is", RMSDtruth)


#%%
#plot ML vs Truth DNB values

#scatterplot
#plotting basics
# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10.0/2, 8.0/2)

plt.figure()
plt.plot(fxx,fyy,'o', color='black')
plt.xlabel('Truth')
plt.ylabel('ML_predicted')
plt.legend()

plt.title('comparison of truth and ML DNB radiances')
plt.show()

#%%

#ERF applications on ML and truth values for image translation

#load raw radiance values (ML and original) or from above unnormPRED unnormTRU
rad=xx #truth DNB values/raw
ML_rad=yy
    
# take the radiances and use the ERF display image scale
Rmax= 1.26e-10 
Rmin=2e-11
    

#ERF image from the paper derived min/max
ERFimage_truth=  255 * np.sqrt((rad[:]-Rmin)/(Rmax - Rmin))
print("this is the max", ERFimage_truth.max(), "this is the min", ERFimage_truth.min())


ERFimage_ML=  255 * np.sqrt((np.abs(ML_rad[:]-Rmin))/(Rmax - Rmin))
print("this is the max", ERFimage_ML.max(), "this is the min", ERFimage_ML.min())

x=ERFimage_truth#[:2000]
y=ERFimage_ML#[:2000]

   
    #%% 
# plotting one image

img = x[0]
imgplot = plt.imshow(img, cmap='gray')
plt.title('ERF imagery truth')
plt.colorbar()
plt.show()

img2 = y[0]
imgplot= plt.imshow(img2, cmap='gray')
plt.title("ERF imagery ML")
plt.colorbar()
plt.show()
#%%

#plot all images?
#def plotit():
for i in range(len(x)):
    x=ERFimage_truth[i]
    y=ERFimage_ML[i]
    a=np.linspace(0,4000,100)
    b=a
    #plt.plot(x,y)
    #plt.show()
    #print ('This is the end of Truth plot using paper full moon max/min')
  
    plt.figure()
    plt.plot(x,y,'o', color='black')
    plt.xlabel('Truth')
    plt.ylabel('ML_predicted')
    plt.plot(a,b, 'r--', label='perfect')
    plt.legend()
    plt.title('comparison of truth and ML DNB ERF scaled BVIs' )
    plt.show()
  
       
    img = x
    imgplot = plt.imshow(img, cmap='gray', vmin =1000, vmax=4000)
  
    plt.title('ERF imagery truth')
    plt.colorbar()
    plt.show()
    
    img2 = y
    imgplot= plt.imshow(img2, cmap='gray', vmin=100, vmax=4000)
    plt.title("ERF imagery ML")
    plt.colorbar()
    plt.show()
    

# ML_DNB = prediction

#histograms of ERF image ML & ERF truth

#%%

# Xtrain =
# Xtest = 
# Ytrain = 
# Ytest = 
                                                                               
# plt.scatter(Xtrain, Ytrain, color='cornflowerblue', label='training data')
# plt.scatter(Xtest, Ytest, color='fuchsia', label = 'testing data')
# plt.xlabel('X value')
# plt.ylabel('Y value')
# plt.legend()
# plt.show()


# # define functional fits
# def my_fit(m,c,x):
#     # m is the slope of the line
#     # c is the y-intercept
#     # x is the values to evaluate
#     # output: y, the evaluated values
#     y = m*x + c
#     return y

# def my_gradLoss(xtrain,ytrain,ypred):
#     n = float(len(xtrain))
#     gradLoss_m = 2/n * sum(xtrain * (ypred-ytrain)) #derivative wrt m
#     gradLoss_c = 2/n * sum((ypred-ytrain)) #derivative wrt c
    
#     return gradLoss_m, gradLoss_c

# # Building the model
# m = np.random.uniform()
# c = np.random.uniform()

# L = 0.0001                                                         # MODIFY: the learning rate - the size of the "step" to take down gradient
# epochs = 40000                                                     # MODIFY: the number of iterations to run over the entire training set 

# errorHistory = np.empty((epochs,))
# # Performing Gradient Descent 
# for i in range(epochs): 

#     Y_pred = my_fit(m,c,Xtrain)                                    # the current predicted value of y
#     gradLoss_m, gradLoss_c = my_gradLoss(Xtrain, Ytrain, Y_pred)   # compute the direction of down gradient of the loss function with respect to the coefficients
#     m = m - L * gradLoss_m                                         # update the slope m
#     c = c - L * gradLoss_c                                         # update the y-intercept c
    
#     errorHistory[i] = 1/float(len(Y))*np.sum((Ytrain - Y_pred)**2)
    
# print('done training')    
# print('')
# print('  slope (m)             y-int (c)')
# print('----------------------------------')
# print (str(np.around(m,5)) + '                 ' + str(np.around(c,5)))

# # Making predictions - FINAL
# Y_pred = my_fit(m,c,Xtrain)
# Y_predTest = my_fit(m,c,Xtest)

# plt.figure()
# plt.scatter(X, Y, color='black', label = 'data') 
# # plt.plot(Xtrain, Y_pred, 'x', color='cornflowerblue', label = 'training data') 
# # plt.plot(Xtest, Y_predTest, 'x', color='fuchsia', label= 'testing data') 

# plt.plot(Xtrain, Y_pred,'-k', label='fit by ANN', linewidth=3) # regression line

# slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(X),np.squeeze(Y))
# plt.plot(X,intercept+X*slope,'--',color = 'red', label = 'LSQ: x vs y', linewidth=1)

# plt.legend()
# plt.show()

# #print the error history
# plt.figure()
# plt.plot(np.arange(0,len(errorHistory)),errorHistory,'.-')
# plt.title('loss function')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()