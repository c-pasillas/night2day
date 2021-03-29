#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:39:14 2021

@author: cpasilla
"""

import h5py
import numpy as np
import shutil
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
from glob import glob
from netCDF4 import Dataset
from satpy import Scene
from satpy import available_readers
from satpy import available_writers
import matplotlib
import math

#TODO check spath and z and
spath= '/zdata2/cpasilla/MARCH2019_ALL/'
z= glob(spath + "RAW_MASTER_ARRAY*")[0]

with open (z, 'rb') as f:
    CASE= np.load(f)
 
#TODO go back and put in full array later   
aa=CASE[:,:,:,:]
 # need a key  for the positions.
 
lat = aa[:,:,:,0]
long = aa[:,:,:,1]
M16 = aa[:,:,:,2]
M15 = aa[:,:,:,3]
M14 = aa[:,:,:,4]
M13 = aa[:,:,:,5]
M12 = aa[:,:,:,6]
DNB = aa[:,:,:,7]   
    
 
print("M16 raw min/max values are", np.amax(M16),"and", np.amin (M16))
print("M15 raw min/max values are", np.amax(M15),"and", np.amin (M15))
print("M14 raw min/max values are", np.amax(M14),"and", np.amin (M14))
print("M13 raw min/max values are", np.amax(M13),"and", np.amin (M13))
print("M12 raw min/max values are", np.amax(M12),"and", np.amin (M12))
print("DNB raw min/max values are", np.amax(DNB),"and", np.amin (DNB))


#use the ATDB values for the max/min of each channel
#BT emissive bands
    #M16 BTrange 190 - 340
    #M15 BTrange 190 - 343
    #M14 BTrange 190 - 336
    #M13 BT range 230-343/343-634
    #M12 BT range 230-353

    
#set max min variable so if I want to adjust these beyond the ATBD 
#do it here and all formulas will check out
M16mn = 190
M16mx = 340 
M15mn = 190
M15mx = 343
M14mn = 190
M14mx = 336
M13mn = 230
M13mx = 634
M12mn = 230
M12mx = 353

#normalization band value formulas
# (Max- value)/(max-min) for channels/bands in LWIR w BTs
# (Value - min)/(max-min) for channel/bands in visible SWIR

# formula for max/min BTDs where A = CH1 B = CH2  where  CH1-CH2
#normalized = (Amax-Bmin)-(Avalue-Bvalue)/(Amax-Bmin)-(Amin-Bmax)

#make normalized variables 
#print both to show the norm is between 0-1 and has clipped all the -9999 data
M16norm = (M16mx-M16[:])/(M16mx-M16mn) 
print('M16norm max min are', np.max(M16norm), np.min(M16norm))
M16norm[M16norm>1]=1
print('M16norm  final max min are', np.max(M16norm), np.min(M16norm))

M15norm = (M15mx-M15[:])/(M15mx-M15mn)
print('M15norm max min are', np.max(M15norm), np.min(M15norm))
M15norm[M15norm>1]=1
print('M15norm  final max min are', np.max(M15norm), np.min(M15norm))

M14norm = (M14mx-M14[:])/(M14mx-M14mn)
print('M14norm max min are', np.max(M14norm), np.min(M14norm))
M14norm[M14norm>1]=1
print('M14norm  final max min are', np.max(M14norm), np.min(M14norm))

M13norm = (M13mx -M13[:])/(M13mx-M13mn)
print('M13norm max min are', np.max(M13norm), np.min(M13norm))
M13norm[M13norm>1]=1
print('M13norm  final max min are', np.max(M13norm), np.min(M13norm))

M12norm =(M12mx-M12[:])/(M12mx-M12mn)
print('M12norm max min are', np.max(M12norm), np.min(M12norm))
M12norm[M12norm>1]=1
print('M12norm  final max min are', np.max(M12norm), np.min(M12norm))

#######  MAKE THE BTDS AND THE NORMALIZED BTD VARIABLES HERE
 
BTD1216 = (M12[:]-M16[:])
BTD1215 = (M12[:]-M15[:])
BTD1316 = (M13[:]-M16[:])
BTD1315 = (M13[:]-M15[:])
BTD1415 = (M14[:]-M15[:])
BTD1416 = (M14[:]-M16[:])
BTD1516 = (M15[:]-M16[:])
 
BTD1216norm = ((M12mx-M16mn)-(M12[:]-M16[:]))/((M12mx-M16mn)-(M12mn-M16mx))
print('BTD1216norm max min are', np.max(BTD1216norm), np.min(BTD1216norm))
BTD1216norm[BTD1216norm>1]=1
print('BTD1216norm  final max min are', np.max(BTD1216norm), np.min(BTD1216norm))

BTD1215norm = ((M12mx-M15mn)-(M12[:]-M15[:]))/((M12mx-M15mn)-(M12mn-M15mx))
print('BTD1215norm max min are', np.max(BTD1215norm), np.min(BTD1215norm))
BTD1215norm[BTD1215norm>1]=1
print('BTD1215norm  final max min are', np.max(BTD1215norm), np.min(BTD1215norm))

BTD1316norm = ((M13mx-M16mn)-(M13[:]-M16[:]))/((M13mx-M16mn)-(M13mn-M16mx))
print('BTD1316norm max min are', np.max(BTD1316norm), np.min(BTD1316norm))
BTD1316norm[BTD1316norm>1]=1
print('BTD1316norm  final max min are', np.max(BTD1316norm), np.min(BTD1316norm))

BTD1315norm = ((M13mx-M15mn)-(M13[:]-M15[:]))/((M13mx-M15mn)-(M13mn-M15mx))
print('BTD1315norm max min are', np.max(BTD1315norm), np.min(BTD1315norm))
BTD1315norm[BTD1315norm>1]=1
print('BTD1315norm  final max min are', np.max(BTD1315norm), np.min(BTD1315norm))

BTD1415norm = ((M14mx-M15mn)-(M14[:]-M15[:]))/((M14mx-M15mn)-(M14mn-M15mx))
print('BTD1415norm max min are', np.max(BTD1415norm), np.min(BTD1415norm))
BTD1415norm[BTD1415norm>1]=1
print('BTD1415norm  final max min are', np.max(BTD1415norm), np.min(BTD1415norm))

BTD1416norm = ((M14mx-M16mn)-(M14[:]-M16[:]))/((M14mx-M16mn)-(M14mn-M16mx))
print('BTD1416norm max min are', np.max(BTD1416norm), np.min(BTD1416norm))
BTD1416norm[BTD1416norm>1]=1
print('BTD1416norm  final max min are', np.max(BTD1416norm), np.min(BTD1416norm))

BTD1516norm = ((M15mx-M16mn)-(M15[:]-M16[:]))/((M15mx-M16mn)-(M15mn-M16mx))
print('BTD1516norm max min are', np.max(BTD1516norm), np.min(BTD1516norm))
BTD1516norm[BTD1516norm>1]=1
print('BTD1516norm  final max min are', np.max(BTD1516norm), np.min(BTD1516norm))


#TODO DNB NORM

#normalize the DNB
# recall that the raw radiance values here were reduced in the satpy process
# as verified in the max/min part need to multiply by e-4 before mathing 
#or adjust the max/min by e-4 and leave as is

DNBfix= DNB[:]* 1e-4
x=DNBfix
xx=np.log10(x)



print("DNB raw max/min values are", np.amax(DNB),"and", np.amin (DNB))
print("DNB fixed max/min values are", np.amax(DNBfix),"and", np.amin (DNBfix))
print("x values max/min are", np.amax(x), "and", np.min(x))
print("xx values max/min are", np.amax(xx), "and", np.min(xx))
#observed radiances in moon  10-6 full moon 10-10 new moon
#specificed range is 3e-9 to 2e-2
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


# (Value - min)/(max-min) for channel/bands in visible SWIR
#normalize DNB to night sensor lims
DNB_norm = (x[:]-DNBmn)/(DNBmx-DNBmn)
print('DNBnorm to night sensor max min are', np.max(DNB_norm), np.min(DNB_norm))
DNB_norm[DNB_norm>1]=1
DNB_norm[DNB_norm<0]=0
print('DNBnorm to night sensor final max min are', np.max(DNB_norm), np.min(DNB_norm))

#normalize to full moon lims per ERF
DNB_fullmoon_norm = (x[:]-DNBmn_fullmoon)/(DNBmx_fullmoon-DNBmn_fullmoon)
print('DNBnorm to full moon ERF lims max min are', np.max(DNB_fullmoon_norm), np.min(DNB_fullmoon_norm))
DNB_fullmoon_norm[DNB_fullmoon_norm>1]=1
DNB_fullmoon_norm[DNB_fullmoon_norm<0]=0
print('DNBnorm to full moon ERF lims final max min are', np.max(DNB_fullmoon_norm), np.min(DNB_fullmoon_norm))

#normalize to new moon lims per ERF
DNB_newmoon_norm = (x[:]-DNBmn_newmoon)/(DNBmx_newmoon-DNBmn_newmoon)
print('DNBnorm to new moon ERF lims max min are', np.max(DNB_newmoon_norm), np.min(DNB_newmoon_norm))
DNB_newmoon_norm[DNB_newmoon_norm>1]=1
DNB_newmoon_norm[DNB_newmoon_norm<0]=0
print('DNBnorm to new moon ERF lims final max min are', np.max(DNB_newmoon_norm), np.min(DNB_newmoon_norm))

#normalize to log of RAW

LDNB_norm = (xx[:]-LDNBmn)/(LDNBmx-LDNBmn)
print('log DNBnorm to night sensor max min are', np.max(LDNB_norm), np.min(LDNB_norm))
LDNB_norm[LDNB_norm>1]=1
LDNB_norm[LDNB_norm<0]=0
print('log DNBnorm to night sensor final max min are', np.max(DNB_norm), np.min(DNB_norm))

# norm to log of ERF full moon lims
LDNB_fullmoon_norm = (xx[:]-LDNBmn_fullmoon)/(LDNBmx_fullmoon-LDNBmn_fullmoon)
print(' log DNBnorm to full moon ERF lims max min are', np.max(LDNB_fullmoon_norm), np.min(LDNB_fullmoon_norm))
LDNB_fullmoon_norm[LDNB_fullmoon_norm>1]=1
LDNB_fullmoon_norm[LDNB_fullmoon_norm<0]=0
print('log DNBnorm to full moon ERF lims final max min are', np.max(LDNB_fullmoon_norm), np.min(LDNB_fullmoon_norm))

#normalize to log of  new moon lims per ERF
LDNB_newmoon_norm = (xx[:]-LDNBmn_newmoon)/(LDNBmx_newmoon-LDNBmn_newmoon)
print(' log DNBnorm to new moon ERF lims max min are', np.max(LDNB_newmoon_norm), np.min(LDNB_newmoon_norm))
LDNB_newmoon_norm[LDNB_newmoon_norm>1]=1
LDNB_newmoon_norm[LDNB_newmoon_norm<0]=0
print('log DNBnorm to new moon ERF lims final max min are', np.max(LDNB_newmoon_norm), np.min(LDNB_newmoon_norm))


# norm to miller recommendations

MillerLDNB_fullmoon_norm = (xx[:]-MillerLDNBmn_fullmoon)/(MillerLDNBmx_fullmoon-MillerLDNBmn_fullmoon)
print(' Miller log DNBnorm to full moon ERF lims max min are', np.max(MillerLDNB_fullmoon_norm), np.min(MillerLDNB_fullmoon_norm))
MillerLDNB_fullmoon_norm[MillerLDNB_fullmoon_norm>1]=1
MillerLDNB_fullmoon_norm[MillerLDNB_fullmoon_norm<0]=0
print('Miller log DNBnorm to full moon ERF lims final max min are', np.max(MillerLDNB_fullmoon_norm), np.min(MillerLDNB_fullmoon_norm))

###############################MATH IS DONE###################
####MAKE A PRODUCT AND FILL ##################################
 
OUTPUTARRAY2 = np.stack((lat,long,M16,M15,M14,M13,M12,DNB,M16norm,M15norm,M14norm,
                         M13norm,M12norm, BTD1216, BTD1215, BTD1316, BTD1315, 
                         BTD1415, BTD1416, BTD1516, BTD1216norm, BTD1215norm, BTD1316norm, 
                         BTD1315norm, BTD1415norm, BTD1416norm, BTD1516norm,DNBfix,DNB_norm,
                         DNB_fullmoon_norm, DNB_newmoon_norm,LDNB_norm,LDNB_fullmoon_norm, 
                         LDNB_newmoon_norm, MillerLDNB_fullmoon_norm), axis=-1)

print('this is the end of master array w M12-M16 RAW, DNB fixed RAW, BTDs raw, M12-16norm, BTDnorm and various DNB norm')

channel_order = ["lat","long","M16","M15","M14","M13","M12","DNB","M16norm","M15norm",
                 "M14norm","M13norm","M12norm", "BTD1216", "BTD1215", "BTD1316",
                 "BTD1315","BTD1415", "BTD1416", "BTD1516", "BTD1216norm", "BTD1215norm",
                 "BTD1316norm","BTD1315norm", "BTD1415norm", "BTD1416norm", "BTD1516norm",
                 "DNBfix","DNB_norm", "DNB_fullmoon_norm", "DNB_newmoon_norm","LDNB_norm",
                 "LDNB_fullmoon_norm", "LDNB_newmoon_norm", "MillerLDNB_fullmoon_norm"]

channeldic={label: position for position,label in enumerate(channel_order)}
import time
timestr = time.strftime("%Y%m%d-%H%M")
np.save(spath + "FINAL_VALID_ARRAY_" + timestr, OUTPUTARRAY2)


with open(spath + "Final_Valid_Channel_Dictionary.txt", 'w') as f:
    f.write(str(channeldic))
    
