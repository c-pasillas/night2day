#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:55:46 2021
#edited paths to account for server locations
#edited 24 feb to account for if there is missing files (ie no DNB or Mband)
#edited 25 feb - commented out to run just the array, need to make processonesampel into 2 fxn
@author: cpasilla
"""

import numpy as np
import os
from collections import defaultdict
from glob import glob
from netCDF4 import Dataset
from satpy import Scene
from satpy import available_readers
from satpy import available_writers
from satpy import resample
import math

#TODO adjust spath and spath 2 before running 
#spath= '/Users/cpasilla/PHD_WORK/CASENAME/'
spath = '/zdata2/cpasilla/MARCH2019_ALL/'

rawpath = spath   + 'RAWDATA/'
spath2= spath +'COLOCATED/RAW_'
multiples=256
#putting included functions first then the main

#this is designed to go through file directory and find the unique times of data 
# aka "samples" that will be used to group the channel data by              
    
#File looks like GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b*
#parts[0] CHANNELS
#parts[1] satellite name
#parts[2] DATE
#parts[3] START TIME
#parts[4] END TIME     
def missing_timesteps(sampleset):
    bad_files =[]
    good_sampleset ={}
    for starttime,my_files in sampleset.items():
        if len(my_files)!=2:
            bad_files.append(starttime)
            print('missing data files for', starttime)
            continue
        else:
            good_sampleset[starttime] = my_files
    print("found", len(bad_files),"bad data sets. go to  badfiles.txt to review")
    with open(spath + "badfiles.txt", "w") as f:
        f.write(str(bad_files))
        f.write('\n')
    return good_sampleset

def find_timesteps():
    paths = glob(rawpath+'G*')
    print(paths)
    sampleset = defaultdict(list)
    for path in paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        starttime = parts[2]+ '_'+ parts[3]
        sampleset[starttime].append(path)
    print ("these are the # of samples by DTG", len(sampleset))
    return missing_timesteps(sampleset)   
      

def inspecth5(sampleset):
    for starttime,my_files in sampleset.items():
#open the files in the dic
        print(" *******************start of inspecting", starttime, "*****************************")
        original_scn = Scene(reader = 'viirs_sdr', filenames=my_files)
        original_scn.load(['M12', 'M13','M14','M15','M16','DNB','dnb_latitude','dnb_longitude', 'm_latitude','m_longitude'])
        print('the channels are loaded')
        print('these are the start shapes of data files')
        print('M12', original_scn['M12'].shape)
        print('M13', original_scn['M13'].shape)
        print('M14', original_scn['M14'].shape)
        print('M15', original_scn['M15'].shape)
        print('M16', original_scn['M16'].shape)
        print('DNB', original_scn['DNB'].shape)
        print('DNB lat', original_scn['dnb_latitude'].shape)
        print('DNB long',original_scn['dnb_longitude'].shape)
        print('M band lat',original_scn['m_latitude'].shape)
        print('M band long',original_scn['m_longitude'].shape)
        print("******************************end inpsection of", starttime, "*******************************")
        print('this is the end of initial shape confirmation')

#if we want to see the original shapes uncomment below line and run
#inspecth5(find_timesteps())
    
# cropping functions


def croptomultiples(casearray):
    rows=casearray.shape[1]
    columns=casearray.shape[2]
    newrows= rows- (rows % multiples)
    newcolumns=columns - ( columns % multiples)
    finalarray=casearray[:, 0:newrows, 0:newcolumns, :]
    return finalarray

def find9999front(arow):
    for i in range(len(arow)):
        x=arow[i]
        if not math.isclose(x,-9999):
            return i
    return 0 
    
def find9999back(arow):
    for i in range(1,len(arow)):
        x=arow[-i]
        if not math.isclose(x,-9999):
            return i
    return 0

def findXD9999(arrayXD):
    front=[]
    back=[]
    for row in arrayXD:
        front.append(find9999front(row))
        back.append(find9999back(row))
        #print(len(front))
    print('the front position is',max(front), 'the back position is', max(back))  
    return max(front), max(back) 
    
def cropdeadspace(single):    
    frontposition=[]
    backposition=[]
    channels= [single[:,:,i] for i in range(0,single.shape[-1])]
    for c in channels:
        maxfront,maxback = findXD9999(c)
        frontposition.append(maxfront)
        backposition.append(maxback)
    maxfrontposition = max(frontposition)
    maxbackposition = max(backposition)
    print('the front position is',maxfrontposition, 'the back position is', maxbackposition) 
    rowlength=single.shape[-2]
    croppedsingle=single[:,maxfrontposition:rowlength-maxbackposition,:]
    return croppedsingle, maxfrontposition, maxbackposition

def finddims(arrays):
    ys=[a.shape[0] for a in arrays]   
    xs=[a.shape[1] for a in arrays]   
    print("Y is", min(ys), max(ys), "X is", min(xs),max (xs))
    return min(ys), max(ys), min(xs),max (xs)

def croprawarray(sampleoutput):
    ymin, ymax, xmin, xmax = finddims(sampleoutput.values())
    FRS={starttime: a[0:ymin,0:xmin] for starttime, a in sampleoutput.items()}
    return FRS

# this is designed to pull out the group of files that occur at the same time 
#"sampleset" to run through the colocation and channel combining process for each timestep
def processonesample(starttime,my_files):
    #original_scn = Scene(reader = 'viirs_sdr', filenames=my_files) #geostationary
    #print(original_scn.keys())
    #print('This is the JPSS channels available', original_scn.available_dataset_names())
    #print('This is the JPSS composites available', original_scn.available_composite_names())

    #load the channels and composites to the "scene"
    #original_scn.load(['M12', 'M13','M14','M15', 'M16','DNB','dnb_latitude','dnb_longitude','m_latitude','m_longitude'])

    #print/confirming shape of the original files before data manipulation 
    #print('these are the start shapes of data files')
    #print('M12', original_scn['M12'].shape)
    #print('M13', original_scn['M13'].shape)
    #print('M14', original_scn['M14'].shape)
    #print('M15', original_scn['M15'].shape)
    #print('M16', original_scn['M16'].shape)
    #print('DNB', original_scn['DNB'].shape)
    #print('DNB lat', original_scn['dnb_latitude'].shape)
    #print('DNB long', original_scn['dnb_longitude'].shape)
    #print(' Mband lat', original_scn['m_latitude'].shape)
    #print(' Mband long', original_scn['m_longitude'].shape)
    #print('this is the end of initial shape confirmation')

    #now resample the Mband to the DNB with a new "Scene" role
    #print('starting resampling of Mbands to DNB')
    #Mband_resample_scn=original_scn.resample(original_scn['DNB'].attrs['area'], resampler='nearest')
    #Mband_resample_scn=resample.resample(

    #print('these are the colocated Mband shapes')
    #print(Mband_resample_scn['M12'].shape)
    #print(Mband_resample_scn['M13'].shape)
    #print(Mband_resample_scn['M14'].shape)
    #print(Mband_resample_scn['M15'].shape)
    #print(Mband_resample_scn['M16'].shape)

    #channels = ['M12','M13','M14','M15','M16','DNB']

    #for c in channels:
       # original_scn.save_dataset(c, f'{spath2}ORIGINAL_{c}_{starttime}.png', writer='simple_image')
         #original_scn.save_dataset(c, spath2 + 'ORIGINAL_' + c + "_" + starttime + '.png', writer='simple_image') #engine='netcdf4')
        #original_scn.save_dataset(c, spath2 + 'ORIGINAL_' + c + "_" + starttime + '.nc', writer='cf')         
       # Mband_resample_scn.save_dataset(c, spath2 + 'COLOCATED_' + c + "_" + starttime + '.png', writer='simple_image') #engine='netcdf4')
       # Mband_resample_scn.save_dataset(c, spath2 + 'COLOCATED_' +  c + "_" + starttime + '.nc', writer='cf')  

    #print('done saving channels')

    
    #####NEED TO MAKE THIS A SEPARATE PART NOT IN processonesample and rename processonesample to onesampleCDF 
    #then a processonesample would run both of these or i would run which i need  onesampleCDF or onesampleARRAY if only need one
    ##SO IF I HAVE TO PROCESS .NC FILES AT DIFFERENT TIMES CUZ MISSING DATA I CAN STILL THEN MAKE THE NEEDED ARRAY
    
#def processonesamplearray (starttime,my_files)
    #upload raw and colocated data files # may not need raw anymore were needed for initial validation of code 
    rawM16=Dataset(spath2 + "ORIGINAL_M16_" + starttime + ".nc")
    rawM15=Dataset(spath2 + "ORIGINAL_M15_" + starttime + ".nc")
    rawM14=Dataset(spath2 + "ORIGINAL_M14_" + starttime + ".nc")
    rawM13=Dataset(spath2 + "ORIGINAL_M13_" + starttime + ".nc")
    rawM12=Dataset(spath2 + "ORIGINAL_M12_" + starttime + ".nc")
    rawDNB=Dataset(spath2 + "ORIGINAL_DNB_" + starttime + ".nc")

    cM16=Dataset(spath2 + "COLOCATED_M16_" + starttime + ".nc")
    cM15=Dataset(spath2 + "COLOCATED_M15_" + starttime + ".nc")
    cM14=Dataset(spath2 + "COLOCATED_M14_" + starttime + ".nc")
    cM13=Dataset(spath2 + "COLOCATED_M13_" + starttime + ".nc")
    cM12=Dataset(spath2 + "COLOCATED_M12_" + starttime + ".nc")
    cDNB=Dataset(spath2 + "COLOCATED_DNB_" + starttime + ".nc")


    # get info on raw data the colocated and original formats
    #Nan--> missing #array.filled(-9999) replaces missing values
    #pull out lat/long
    #fill all the arrays 
    filleddic = {}

    filleddic['FLat'] =  cDNB['latitude'][:].filled(-9999)
    filleddic['FLong'] = cDNB['longitude'][:].filled(-9999)

    filleddic['FrawM16'] = rawM16['M16'][:].filled(-9999)
    filleddic['FrawM15'] = rawM15['M15'][:].filled(-9999)
    filleddic['FrawM14'] = rawM14['M14'][:].filled(-9999)
    filleddic['FrawM13'] = rawM13['M13'][:].filled(-9999)
    filleddic['FrawM12'] = rawM12['M12'][:].filled(-9999)
    filleddic['FrawDNB'] = rawDNB['DNB'][:].filled(-9999)

    filleddic['FcM16'] = cM16['M16'][:].filled(-9999)
    filleddic['FcM15'] = cM15['M15'][:].filled(-9999)
    filleddic['FcM14'] = cM14['M14'][:].filled(-9999)
    filleddic['FcM13'] = cM13['M13'][:].filled(-9999)
    filleddic['FcM12'] = cM12['M12'][:].filled(-9999)
    filleddic['FcDNB'] = cDNB['DNB'][:].filled(-9999)

    #print initial max/min values before the -9999

    for name, array in filleddic.items(): 
        print (f'{name} max is {np.max(array)} and min is {np.min(array)}')

    for name, array in filleddic.items():       
        Rma=array[array != -9999]
        Rma.max()
        Rma.min()
        A=(array>0).sum()
        N=(array==-9999).sum()
        D=N+A

        print(name+ "max is", np.max(Rma), "min is", np.min(Rma))
        print(name + "has this many valid", A,)
        print(name + "has this many null values", N,)
        print(name + "has this many  values", D,)

    print("done looking at values, check order of magnitude raw DNB max/min is",
          np.max(filleddic['FrawDNB']), np.min(filleddic['FrawDNB']), 
          "colocated DNB max/min os",
          np.max(filleddic['FcDNB']), np.min(filleddic['FcDNB']))  

    print('combine to a single 3D array')

    for name, array in filleddic.items():   
        print (f'{name} shape is {array.shape}') 
    # we have dic of numpy arrays for each colocated file

    global colocateddic
    colocateddic = {}
    for name, array in filleddic.items(): 
        if "raw" not in name:
            colocateddic[name] = array
    print('colocated dictionary keys are', colocateddic.keys())

    dicorder=['FLat', 'FLong', 'FcM16', 'FcM15', 'FcM14', 'FcM13', 'FcM12', 'FcDNB']
    arrays=[colocateddic[k] for k in dicorder]
    single=np.stack(tuple(arrays),axis=-1) #combines
    print(single.shape)
    nodeadspacesingle, cropleft, cropright= cropdeadspace(single)
    return nodeadspacesingle


#processes all the samples in the case
def processallsamples(sampleset,limit=None):
    if limit is None:
        limit=len(sampleset)
    global sampleoutput
    sampleoutput={}
    for starttime,myfiles in sampleset.items():
        if limit == 0:
            break
        else:
            limit = limit-1
        print( (len(sampleset)-limit),"/",len(sampleset))
        print("******************started", starttime, "*******************************")
          #for i in len(sampleset):
        if len(myfiles)!=2:
            print('missing data files for', starttime)
            continue
        else:
        #open the files in the dic
            x = processonesample(starttime,myfiles)# NODEADSPACESINGLE IS THIS X
            sampleoutput[starttime] = x
        print("***************completed processing of", starttime, "*******************************")
    #crops to the smallest
    CROPPEDdict=croprawarray(sampleoutput)  
    #stack them
    SINGLECASEARRAY=np.stack(tuple(CROPPEDdict.values())) #combines
    print(SINGLECASEARRAY.shape)
    # now reduce to the 256x256y size
    return croptomultiples(SINGLECASEARRAY), CROPPEDdict   

def main():
    SINGLE_CASE_ARRAY, CROPPEDdict = processallsamples(find_timesteps())#, limit=3)
#save
    import time
    timestr = time.strftime("%Y%m%d-%H%M")
    np.save(spath + "RAW_MASTER_ARRAY_" + timestr, SINGLE_CASE_ARRAY)
#array colocated to DNB, with no -9999 edges, all same size to smallest dims, all channles,all samples



if __name__ == "__main__":
    main()
    #find_timesteps()


