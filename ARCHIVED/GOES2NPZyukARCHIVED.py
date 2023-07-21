#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import datetime as date
from pathlib import Path

#TODO adjust spath and spath 2 before running 
#spath= '/Users/cpasilla/PHD_WORK/CASENAME/'
spath = '/gdata5/cpasilla/GOESABICOLOCATION_TEST/GOES'

rawpath = spath
spath2= spath +'COLOCATED/RAW_'

#this is designed to go through file directory and find the unique times of data 
# aka "samples" that will be used to group the channel data by              
    
#parts[0] GEO
#parts[1] Channel
#parts[2] satellite
#parts[3] START TIME
#parts[4] END TIME    
#OR_ABI-L1b-RadF-M6C07_G17_s20201560830320_e20201560839398_c20201560839443.nc


def missing_timesteps(sampleset):
    bad_files =[]
    good_sampleset ={}
    for starttime,my_files in sampleset.items():
        if len(my_files)!=4:
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
    paths = glob(path+'O*')
    print(paths)
    sampleset = defaultdict(list)
    for path in paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        starttime = parts[3]
        sampleset[starttime].append(path)
    print ("these are the # of samples by DTG", len(sampleset))
    return missing_timesteps(sampleset)   
      
    
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
   
    
#def processonesamplearray (starttime,my_files)
    #upload ABI data files 
    rawC07=Dataset(spath2 + "G*C07*" + starttime + "*.nc")
    rawC11=Dataset(spath2 + "G*C11*" + starttime + "*.nc")
    rawC13=Dataset(spath2 + "G*C13*" + starttime + "*.nc")
    rawC15=Dataset(spath2 + "G*C15*" + starttime + "*.nc")
   

    # get info on raw data the colocated and original formats
    #Nan--> missing #array.filled(-9999) replaces missing values
    #pull out lat/long
    #fill all the arrays 
    filleddic = {}

    filleddic['FLat'] =  rawC07['latitude'][:].filled(-9999)
    filleddic['FLong'] = rawC07['longitude'][:].filled(-9999)

    filleddic['FrawC07'] = rawC07['C07'][:].filled(-9999)
    filleddic['FrawC11'] = rawC11['C11'][:].filled(-9999)
    filleddic['FrawC13'] = rawC13['C13'][:].filled(-9999)
    filleddic['FrawC15'] = rawC15['C15'][:].filled(-9999)

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

    dicorder=['FLat', 'FLong', 'FrawC07', 'FrawC11', 'FrawC13', 'FrawC15'] 
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
        if len(myfiles)!=4:
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

def main(args): 
    path = Path(args.abi_dir).resolve()
    print('path is', path)
    SINGLE_CASE_ARRAY, CROPPEDdict = processallsamples(find_timesteps())#, limit=3)
#save
    import time
    timestr = time.strftime("%Y%m%d-%H%M")
    np.save(path + "RAW_MASTER_ARRAY_GOESONLY" + timestr, SINGLE_CASE_ARRAY)
#array colocated to DNB, with no -9999 edges, all same size to smallest dims, all channles,all samples



#if __name__ == "__main__":
 #   main()
    #find_timesteps()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import datetime as date
from pathlib import Path


#path = Path(args.abi_dir).resolve()
#print('path is', path)
#spath = '/zdata2/cpasilla/MARCH2019_ALL/'

#rawpath = spath   + 'RAWDATA/'
#spath2= spath +'COLOCATED/RAW_'

#this is designed to go through file directory and find the unique times of data 
# aka "samples" that will be used to group the channel data by              
    
#parts[0] GEO
#parts[1] Channel
#parts[2] satellite
#parts[3] START TIME
#parts[4] END TIME    
#OR_ABI-L1b-RadF-M6C07_G17_s20201560830320_e20201560839398_c20201560839443.nc


def missing_timesteps(sampleset):
    bad_files =[]
    good_sampleset ={}
    for starttime,my_files in sampleset.items():
        if len(my_files)!=4:
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

def find_timesteps(path):
    paths = glob(path/f'{path.name}O*')
      #filename = path / f'{path.name}_J01_RefALL_case.npz'
    print(paths)
    sampleset = defaultdict(list)
    for path in paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        starttime = parts[3]
        sampleset[starttime].append(path)
    print ("these are the # of samples by DTG", len(sampleset))
    return missing_timesteps(sampleset)   
      
    
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
   
    
#def processonesamplearray (starttime,my_files)
    #upload ABI data files 
    rawC07=Dataset(spath2 + "G*C07*" + starttime + "*.nc")
    rawC11=Dataset(spath2 + "G*C11*" + starttime + "*.nc")
    rawC13=Dataset(spath2 + "G*C13*" + starttime + "*.nc")
    rawC15=Dataset(spath2 + "G*C15*" + starttime + "*.nc")
   

    # get info on raw data the colocated and original formats
    #Nan--> missing #array.filled(-9999) replaces missing values
    #pull out lat/long
    #fill all the arrays 
    filleddic = {}

    filleddic['FLat'] =  rawC07['latitude'][:].filled(-9999)
    filleddic['FLong'] = rawC07['longitude'][:].filled(-9999)

    filleddic['FrawC07'] = rawC07['C07'][:].filled(-9999)
    filleddic['FrawC11'] = rawC11['C11'][:].filled(-9999)
    filleddic['FrawC13'] = rawC13['C13'][:].filled(-9999)
    filleddic['FrawC15'] = rawC15['C15'][:].filled(-9999)

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

    dicorder=['FLat', 'FLong', 'FrawC07', 'FrawC11', 'FrawC13', 'FrawC15'] 
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
        if len(myfiles)!=4:
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

def main(args): 
    path = Path(args.abi_dir).resolve()
    print('path is', path)
    SINGLE_CASE_ARRAY, CROPPEDdict = processallsamples(find_timesteps(path))#, limit=3)
#save
    import time
    timestr = time.strftime("%Y%m%d-%H%M")
    np.save(path + "RAW_MASTER_ARRAY_GOESONLY" + timestr, SINGLE_CASE_ARRAY)
#array colocated to DNB, with no -9999 edges, all same size to smallest dims, all channles,all samples



#if __name__ == "__main__":
 #   main(args.abi_dir)
    #find_timesteps()
    

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
    
    

