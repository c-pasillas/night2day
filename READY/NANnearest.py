import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

#putting this work on stand by because its only needed for end data.  we want to test to see if model fit breaks w some nans or just leaves the nans there or interpolates.  The answer to this determines IF we need ot go back and work on this situation or not

def count_nan(array):
    return np.sum(np.isnan(array))
    
def is_all_nan(array):
    return all(np.isnan(array))

def find_nan_row(array):
    for i,row in enumerate(array):
        if is_all_nan(row):
            for j in range (i+1,len(array)):
                if not is_all_nan(array[j]):
                    return(i-1, j)
    return(0,0)     
    
def fill_in_nan_row(array, start, stop):
    a=(stop-start)//2
    for i in range(start + 1, start + a):
        array[i,:]=array[start,:]
    for i in range(start + a, stop):
        array[i,:]=array[stop,:]
            
def fill_in_nan_array(case, channels):
    log.info(f'filling in nan arrays for {channels}')
    for channel in channels:
        log.info(f'filling in nan arrays for {channel}')
        images = case[channel]   
        for i,image in enumerate(images):
            log.info(f'filling in image {i+1} / {len(images)}')
            start,stop = find_nan_row(image)
            if start != 0:
                fill_in_nan_row(image, start,stop)
        images[np.isnan(images)] = np.nanmean(images)





def has_nans(arr, i):
    has_nan= np.isnan(arr).any() 
    if has_nan:
        print(f"I found a NAN at {i}")
    return has_nan

def remove_nans(case):
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for NANs")
    IDX = [i for i in range(len(case_4D)) if not has_nans(case_4D[i],i)]  
    new_case_4D = case_4D[IDX]
    new_samples = case['samples'][IDX]
    new_patch = case['PATCH_ID'][IDX]
    
    nan_nearest = {"PATCH_ID": new_patch, "samples": new_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        nan_nearest[channels[i]] = new_case_4D[:,:,:,i] 
    return nan_nearest
    
def NAN(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    nan_nearest = remove_nans(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_nearestNAN.npz"
    np.savez_compressed(savepath,**nana_nearest )