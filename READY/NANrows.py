import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

def has_many_nans(arr, i):
    count_nan= np.count_nonzero(np.isnan(arr)) 
    threshold = arr.shape[1]
    print(f'I am in sample {i} and have counted {count_nan} nans and threshold is {threshold}')
    return count_nan >= threshold

def remove_nans(case):
    #convert from the 3D dict of arrays to a 4D array (takes the longest time is there something better?)
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for NANs")
    #filter for nans
    IDX = [i for i in range(len(case_4D)) if not has_many_nans(case_4D[i],i)] 
    print(f"im in remove nans and we have kept {len(IDX)}/{len(case_4D)} files")
    new_case_4D = case_4D[IDX]
    new_case_4D = np.nan_to_num(new_case_4D, copy=False, nan=0)
    new_samples = np.array(case['samples'])[IDX]
    nanned = {"samples": new_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        nanned[channels[i]] = new_case_4D[:,:,:,i] 
    return nanned # this returns a set with no NAN rows


def keep_nans(case):
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for NANs")
    #filter for NAN rows, keep ones with NANrows
    IDX = [i for i in range(len(case_4D)) if has_many_nans(case_4D[i],i)] 
    print(f"im in keep nans and we have kept {len(IDX)}/{len(case_4D)} files")
    new_case_4D = case_4D[IDX]
    new_samples = np.array(case['samples'])[IDX]
    nanned = { "samples": new_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        nanned[channels[i]] = new_case_4D[:,:,:,i] 
    return nanned


def both_nans(case):
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for NANs")
    #filter for NAN rows, keep ones with NANrows
    
    #make the keep_nanrows file
    IDX = [i for i in range(len(case_4D)) if has_many_nans(case_4D[i],i)] 
    print(f"im in keep nans and we have kept {len(IDX)}/{len(case_4D)} files")
    new_case_4D = case_4D[IDX]
    new_samples = np.array(case['samples'])[IDX]
    keepnanned = { "samples": new_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        keepnanned[channels[i]] = new_case_4D[:,:,:,i] 
    
    #make the remove nanrows file
    notIDX = [i for i in range(len(case_4D)) if i not in IDX] 
    print(f"im in remove nans and we have kept {len(notIDX)}/{len(case_4D)} files")
    Rnew_case_4D = case_4D[notIDX]
    Rnew_case_4D = np.nan_to_num(Rnew_case_4D, copy=False, nan=0)#TODO better nan policy?
    Rnew_samples = np.array(case['samples'])[notIDX]
    removenanned = {"samples": Rnew_samples, "channels": channels}
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        removenanned[channels[i]] = Rnew_case_4D[:,:,:,i] 
    
    return keepnanned, removenanned  
    
def NANcase(case):
    newcase = remove_nans(case)
    return newcase   
    
    
def NAN(args):
    print("I am about to load the case and args are" , args)
    case = np.load(args.npz_path)
    print("I loaded the case")
    if args.both:
        keepnanned, removenanned = both_nans(case)
        savepath = args.npz_path[:-4] + "_yes_NANrows.npz"
        np.savez_compressed(savepath,**keepnanned) 
        savepath = args.npz_path[:-4] + "_no_NANrows.npz"
        np.savez_compressed(savepath,**removenanned )
        return
    if not args.keep:
        nanned = remove_nans(case)
        end = "_no_NANrows.npz"
    else:
        nanned = keep_nans(case)
        end = "_yes_NANrows.npz"
    print("I am now saving case")
    
    savepath = args.npz_path[:-4] + end
    np.savez_compressed(savepath,**nanned )