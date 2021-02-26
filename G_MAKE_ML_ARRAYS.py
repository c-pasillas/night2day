#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:49:38 2021

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

#TODO  edit paths
spath= '/zdata2/cpasilla/MARCH2019_ALL/'
FVAS= glob(spath + "FINAL_VALID_ARRAY_*")[0]
CD = "Final_Valid_Channel_Dictionary.txt"
NL = 45  #PAOI 45
SL = -5 #PAOI -5
EL = -125 #PAOI -145
WL = 160 #PAOI 170
patchdim = 256

def readdata():
    with open (spath + CD, 'r') as f:
        contents = f.read()
        Channel_Dictionary = eval(contents)
    with open (FVAS, 'rb') as f:
        MLcase= np.load(f)   
        print("case shape is", MLcase.shape)
    return MLcase, Channel_Dictionary
   
#reshape to make sample batches
def imagepatches(originalarray, patchdim):
    d=patchdim
    patches=[]
    for i in range(0,originalarray.shape[0],d):
        for j in range(0,originalarray.shape[1],d):
            p=originalarray[i:i+d, j:j+d]
            patches.append(p)
    return patches    
        
def casepatches(fullcasearray, patchdim):
    casepatches = []
    for i in range(len(fullcasearray)):
        patchlist=imagepatches(fullcasearray[i], patchdim)
        casepatches.extend(patchlist)
    casearray=np.array(casepatches)
    print('the new output shape is', casearray.shape)
    return casearray    
               
#remove non AOI samples
def is_point_in_box(lat,long):
    latok = SL <= lat <= NL
    if WL < EL:
        longok = WL <= long <= EL
    else:
        longok = long >= WL or long <= EL
    return latok and longok
    
def is_in_AOI(image):
    mylist = [(0,0), (-1,0), (0,-1), (-1,-1)]
    for coord in mylist:
        pixel = image[coord]
        lat,long = pixel[0],pixel[1]
        if not is_point_in_box(lat,long):
            print("im not in AOI", lat, long)
            return False
    return True
  
def caseinAOI(casepatches):
    goodcases= []
    for patch in casepatches:
        if is_in_AOI(patch):
            goodcases.append(patch)
    goodcasearray=np.array(goodcases)
    print("the new cases array is", goodcasearray.shape)        
    return goodcasearray
  
#shuffle 
def shuffled(array):
    x=array.copy()
    np.random.shuffle(x)
    return x

######make the ML files

def Predictorarray(array, channels, Channel_Dictionary):
    listchannels = []
    for c in channels:
        position = Channel_Dictionary[c]
        channelarray =array[:,:,:,position]
        listchannels.append(channelarray)
    inputarray=np.stack(listchannels, axis = -1)
    return inputarray

def prepare(array, predictors, truth, Channel_Dictionary):
    datadict={}
    datadict["Xdata"]= Predictorarray(array, predictors, Channel_Dictionary)
    datadict['Ydata']= array[:,:,:,Channel_Dictionary[truth]]
    datadict['Lat']= array[:,:,:,Channel_Dictionary['lat']]
    datadict['Long']= array[:,:,:,Channel_Dictionary['long']]
    return datadict
    
def splitdata(array):
    i = int(len(array)*0.2)
    TestArray = array[:i]
    TrainArray = array[i:,:,:,:]
    return TestArray, TrainArray


# main helper function
def validate_channel_input(predictorstring, channel_dictionary):
    predictorlist=predictorstring.split()
    repeats = []
    nondict = []
    good=set()
    for channel in predictorlist:
        if channel not in channel_dictionary:
            nondict.append(channel)
        elif channel in good:
            repeats.append(channel)
        else:
            good.add(channel)
    if not repeats and not nondict:
        return good
    else:    
        raise ValueError(f'repeated channels are {repeats} and invalid channels are {nondict}')        

def userchannelinput(channel_dictionary):
    while True:
        try:
            print("Show avaliable channels", list(channel_dictionary))
            val = input("Which channels for predictors? ")
            GP=validate_channel_input(val, channel_dictionary)
        
            val2 = input("Which channels for predictand? ")
            if val2 not in channel_dictionary:
                raise ValueError("bad channel name")
            if val2 in GP:
                raise ValueError("bad predictor name")    
            return GP, val2
        except ValueError:
            print('try new input')

def makeMLfiles():
    MLcase, channel_dictionary = readdata()
    patches = casepatches(MLcase, patchdim)
    AOIcases = caseinAOI(patches)
    AOIshuff = shuffled(AOIcases)
    Testdata, Traindata = splitdata(AOIshuff)
    Predictors, TruthC = userchannelinput(channel_dictionary)
    Test = prepare(Testdata, Predictors, TruthC, channel_dictionary)
    Train = prepare(Traindata, Predictors,TruthC, channel_dictionary)
    channeldic={label: position for position,label in enumerate(Predictors)}  
    return Test, Train, channeldic, TruthC
    #save the predictor "XData_test" "XData_train" array and predicand "YData_test", 
    #"YData_train" in a folder with name # that includes the predictor/predictand variables


def makeMLfoldername(channeldic, TruthC):
    Predictors= "_".join(channeldic.keys())
    return f'{TruthC}_Predictors_{Predictors}'
    

def main():
    Test, Train, channeldic, TruthC = makeMLfiles()
    MLpath = spath + "ML_INPUTS/"
    folder = makeMLfoldername(channeldic, TruthC)
    ML_folder = MLpath + folder
    if not os.path.exists(ML_folder):
        os.makedirs(ML_folder)
    
    print('Saving data to files in folder', ML_folder ) 
    print("dimensions are", Test['Xdata'].shape, "and", Train['Xdata'].shape)
    with open(ML_folder + "/Channel_Dictionary.txt", 'w') as f:
        f.write(str([TruthC, channeldic]))
    np.savez(ML_folder + "/data.npz" , Xdata_train=Train['Xdata'], Ydata_train=Train['Ydata'],
       Xdata_test=Test['Xdata'], Ydata_test=Test['Ydata'],
       Lat_train=Train['Lat'], Lon_train=Train['Long'],
       Lat_test=Test['Lat'], Lon_test=Test['Long'] )


main()
 
 





