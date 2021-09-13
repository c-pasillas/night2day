import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

def has_nans(arr, i):
    has_nan= np.isnan(arr).any() 
    if has_nan:
        print(f"I found a NAN at {i}")
    return has_nan

def has_9999s(arr,i):
    has_9999 = np.argwhere(arr == 9999)
    if has_9999:
        print(f"I found a 9999 at {i}")
    return has_9999

def remove_9999s(case):
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for 9999s")
    IDX = [i for i in range(len(case_4D)) if not has_9999s(case_4D[i],i)]  
    new_case_4D = case_4D[IDX]
    new_samples = case['samples'][IDX]
    new_patch = case['PATCH_ID'][IDX]
    
    n9999 = {"PATCH_ID": new_patch, "samples": new_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        n9999[channels[i]] = new_case_4D[:,:,:,i] 
    return n9999


def remove_nans(case):
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for NANs")
    IDX = [i for i in range(len(case_4D)) if not has_nans(case_4D[i],i)]  
    new_case_4D = case_4D[IDX]
    new_samples = case['samples'][IDX]
    new_patch = case['PATCH_ID'][IDX]
    
    nanned = {"PATCH_ID": new_patch, "samples": new_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        nanned[channels[i]] = new_case_4D[:,:,:,i] 
    return nanned
  
def NAN9999(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    n9999 = remove_9999s(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "_NANis9999.npz"
    np.savez(savepath,**n9999 )
    
def NAN(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    nanned = remove_nans(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "_noNAN.npz"
    np.savez(savepath,**nanned )