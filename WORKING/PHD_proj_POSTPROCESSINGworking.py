#PHD_proj_POSTPROCESS
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:00:25 2020
#edited 6 JAN 2021
#edited 23 Feb 2021.  must load all model metric for now.
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
import matplotlib as mpl
import math

spath = "/Users/cpasilla/PHD_WORK/CASENAME/ML_INPUTS/SERVERCASE/"

#load the data and model
data = np.load(spath + 'data.npz') #"data_file"
model = load_model(spath + 'model_C1_UNET_blocks_3_epochs_50.h5', custom_objects = { \
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
prediction= model.predict(data['Xdata_test'])
truth = data['Ydata_test']



PRED=prediction.flatten()
TRU=truth.flatten()


#renormalize data ( raw and ML outputs)

#various truth options (normalized)
#DNB_norm = TRU
DNB_fullmoon_norm =truth
# DNB_newmoon_norm
# LDNB_norm
# LDNB_fullmoon_norm 
# LDNB_newmoon_norm
# MillerLDNB_fullmoon_norm
#ML_DNB_norm = PRED
ML_DNB_fullmoon_norm = prediction
# ML_DNB_newmoon_norm
# ML_LDNB_norm
# ML_LDNB_fullmoon_norm 
# ML_LDNB_newmoon_norm
# ML_MillerLDNB_fullmoon_norm

# max mins
DNBmx = 3E-7 # night sensor
DNBmn = 2E-10 # night sensor

DNBmx_fullmoon = 1e-7 #curtis paper and verbal
DNBmn_fullmoon = 1.26e-10 #curtis paper and verbal

DNBmx_newmoon = 1e-9 # curtis paper and verbal
DNBmn_newmoon = 2e-11# curtis paper and verbal

LDNBmx = math.log10(DNBmx)
LDNBmn = math.log10(DNBmn)

LDNBmx_fullmoon = math.log10(DNBmx_fullmoon)
LDNBmn_fullmoon = math.log10(DNBmn_fullmoon)

LDNBmx_newmoon = math.log10(DNBmx_newmoon)
LDNBmn_newmoon = math.log10(DNBmn_newmoon)

MillerLDNBmx_fullmoon = -8.0
MillerLDNBmn_fullmoon= -9.5


#reverse norm original/truth data values based on what was used for truth 
# #reverse normalize DNB to night sensor lims
# DNB_norm = (x[:]-DNBmn)/(DNBmx-DNBmn)

#RadianceDNB_norm = ((DNBmx-DNBmn)*DNB_norm[:]) + DNBmn
#print('raw radiance from DNBnorm to night sensor max min are', np.max(RadianceDNB_norm), np.min(RadianceDNB_norm))


# #renormalize to full moon lims per ERF
#DNB_fullmoon_norm = (x[:]-DNBmn_fullmoon)/(DNBmx_fullmoon-DNBmn_fullmoon)
Radiance_DNB_fullmoon_norm = ((DNBmx_fullmoon-DNBmn_fullmoon)*DNB_fullmoon_norm[:]) + DNBmn_fullmoon
print('raw radiance from DNBnorm to full moon ERF lims max min are', np.max(Radiance_DNB_fullmoon_norm), np.min(Radiance_DNB_fullmoon_norm))


# renormalize to new moon lims per ERF
# DNB_newmoon_norm = (x[:]-DNBmn_newmoon)/(DNBmx_newmoon-DNBmn_newmoon)
#RadianceDNB_newmoon_norm = ((DNBmx_newmoon-DNBmn_newmoon)*DNB_newmoon_norm[:]) + DNBmn_newmoon
#print('raw radiance from DNBnorm to full moon ERF lims max min are', np.max(RadianceDNB_newmoon_norm), np.min(RadianceDNB_newmoon_norm))

#reverse norm ML data values
# use formula based on which version of TRUTH you used
#make some code or input so knows which one
# #reverse normalize DNB to night sensor lims
# DNB_norm = (x[:]-DNBmn)/(DNBmx-DNBmn)

#ML_RadianceDNB_norm = ((DNBmx-DNBmn)*ML_DNB_norm[:]) + DNBmn
#print('raw radiance from DNBnorm to night sensor max min are', np.max(ML_RadianceDNB_norm), np.min(ML_RadianceDNB_norm))


# #renormalize to full moon lims per ERF
#DNB_fullmoon_norm = (x[:]-DNBmn_fullmoon)/(DNBmx_fullmoon-DNBmn_fullmoon)
ML_Radiance_DNB_fullmoon_norm = ((DNBmx_fullmoon-DNBmn_fullmoon)*ML_DNB_fullmoon_norm[:]) + DNBmn_fullmoon
print('raw radiance from DNBnorm to full moon ERF lims max min are', np.max(ML_Radiance_DNB_fullmoon_norm), np.min(ML_Radiance_DNB_fullmoon_norm))


# renormalize to new moon lims per ERF
# DNB_newmoon_norm = (x[:]-DNBmn_newmoon)/(DNBmx_newmoon-DNBmn_newmoon)
#ML_RadianceDNB_newmoon_norm = ((DNBmx_newmoon-DNBmn_newmoon)*ML_DNB_newmoon_norm[:]) + DNBmn_newmoon
#print('raw radiance from DNBnorm to full moon ERF lims max min are', np.max(ML_RadianceDNB_newmoon_norm), np.min(ML_RadianceDNB_newmoon_norm))

# #normalize to log of RAW
#check why it doesnt like np.log10(x) as need the xx for the logs
# DNBfix= DNB[:]* 1e-4
# x=DNBfix
# xx=np.log10(x)

# LDNB_norm = (xx[:]-LDNBmn)/(LDNBmx-LDNBmn)
# print('log DNBnorm to night sensor max min are', np.max(LDNB_norm), np.min(LDNB_norm))


# # norm to log of ERF full moon lims
# LDNB_fullmoon_norm = (xx[:]-LDNBmn_fullmoon)/(LDNBmx_fullmoon-LDNBmn_fullmoon)
# print(' log DNBnorm to full moon ERF lims max min are', np.max(LDNB_fullmoon_norm), np.min(LDNB_fullmoon_norm))


# #normalize to log of  new moon lims per ERF
# LDNB_newmoon_norm = (xx[:]-LDNBmn_newmoon)/(LDNBmx_newmoon-LDNBmn_newmoon)
# print(' log DNBnorm to new moon ERF lims max min are', np.max(LDNB_newmoon_norm), np.min(LDNB_newmoon_norm))


# # norm to miller recommendations

# MillerLDNB_fullmoon_norm = (xx[:]-MillerLDNBmn_fullmoon)/(MillerLDNBmx_fullmoon-MillerLDNBmn_fullmoon)
# print(' Miller log DNBnorm to full moon ERF lims max min are', np.max(MillerLDNB_fullmoon_norm), np.min(MillerLDNB_fullmoon_norm))




#possible outputs from renorm (raw radiances in theory)
#Radiance_DNB_norm = PRED
# Radiance_DNB_fullmoon_norm 
# Radiance_DNB_newmoon_norm
# Radiance_LDNB_norm
# Radiance_LDNB_fullmoon_norm 
# Radiance_LDNB_newmoon_norm
# Radiance_MillerLDNB_fullmoon_norm
#ML_Radiance_DNB_norm = TRU
# ML_Radiance_DNB_fullmoon_norm 
# ML_Radiance_DNB_newmoon_norm
# ML_Radiance_LDNB_norm
# ML_Radiance_LDNB_fullmoon_norm 
# ML_Radiance_LDNB_newmoon_norm
# ML_Radiance_MillerLDNB_fullmoon_norm



#comparisons
xx=Radiance_DNB_fullmoon_norm
yy=ML_Radiance_DNB_fullmoon_norm

print( 'DNB raw truth max is', np.max(yy))
print( 'DNB raw truth min is', np.min (yy))
print( 'ML raw rad max is', np.max(xx))
print( 'ML raw rad min is', np.min(xx))


#calculate RMSE
from sklearn.metrics import mean_squared_error 
from math import sqrt

RMSDtruth =sqrt(mean_squared_error(PRED,TRU))
print("the real RMSE is", RMSDtruth)


#calculate R^2



#calculate SSIM



#plot ML vs Truth DNB values

#scatterplot
#plotting basics
# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10.0/2, 8.0/2)

plt.figure()
plt.plot(TRU,PRED,'o', color='black')
plt.xlabel('Truth')
plt.ylabel('ML_predicted')
plt.legend()

plt.title('comparison of truth and ML DNB')
plt.show()



#ERF applications on ML and truth values for image translation

#load raw radiance values (ML and original) or from above unnormPRED unnormTRU
rad=Radiance_DNB_fullmoon_norm #truth DNB values/raw
ML_rad=ML_Radiance_DNB_fullmoon_norm 
    
# take the radiances and use the ERF display image scale
Rmax= 1.26e-10 
Rmin=2e-11
    

#ERF image from the paper derived min/max
ERFimage_truth=  255 * np.sqrt((rad[:]-Rmin)/(Rmax - Rmin))
print("this is the max", ERFimage_truth.max(), "this is the min", ERFimage_truth.min())


ERFimage_ML=  255 * np.sqrt((ML_rad[:]-Rmin)/(Rmax - Rmin))
print("this is the max", ERFimage_ML.max(), "this is the min", ERFimage_ML.min())

x=ERFimage_truth[0]
y=ERFimage_ML[0]
plt.plot(x,y)
plt.show()
print ('This is the end of Truth plot using paper full moon max/min')
   
    
    
    # plotting

img = x
imgplot = plt.imshow(img, cmap='gray')
plt.title('ERF imagery truth')
plt.colorbar()
plt.show()

img2 = y
imgplot= plt.imshow(img2, cmap='gray')
plt.title("ERF imagery ML")
plt.colorbar()
plt.show()
#%%

#plot all images?
#def plotit():
for i in range(0,14):
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
    imgplot= plt.imshow(img2, cmap='gray', vmin=1000, vmax=4000)
    plt.title("ERF imagery ML")
    plt.colorbar()
    plt.show()
    
#save the scatterplots 
## save the ML_DNB_rad and the DNB_rad and lat/long for future work

# ML_DNB = prediction