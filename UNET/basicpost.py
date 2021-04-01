#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:00:25 2020

@author: cpasilla
"""

##### post process applications


####using the model make the predictands
#from netCDF4 import Dataset
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import numpy as np
from keras.models import load_model
#data_file = Dataset(') #channels
#model_file = (')#ML model
#truth_file = () #DNB test data

data = np.load('/zdata2/cpasilla/TEST_data.npz') #"data_file"
model = load_model('/zdata2/cpasilla/JAN2020_ALL/OUTPUT/MODEL/model_C1_UNET_blocks_3_epochs_50.h5') # "model_file"
prediction = model.predict(data['Xdata_test'])
truth = data['Ydata_test']


###### load DNB radiances (Ydata_test) for the same set as Xdata_test  
# scatterplot 
import matplotlib as plt
x=prediction
y=truth

plt.plot(x, y, 'o', color='black')
plt.show()
#calculate RMSE
from sklearn.metrics import mean_squared_error 
from math import sqrt

RMSD = sqrt(mean_squared_error(y,x))

print(RMSD)

