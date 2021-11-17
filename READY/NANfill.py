import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold


def fillit(case):  
    #convert from the 3D dict of arrays to a 4D array
    channels = case['channels']
    case_4D = np.stack([case[label] for label in channels], axis = -1)
    print("done stacking case about to filter for NANs")
    #count the NANs
    COUNT1 = np.count_nonzero(np.isnan(case_4D))
    print(f"there are {COUNT1} NANs")
    ##fill it
    new_case_4D = np.nan_to_num(case_4D, copy=True, nan=9999, posinf=None, neginf=None)
    filled = {"channels": channels}
    
    ###count the NANs
    COUNT2 = np.count_nonzero(np.isnan(new_case_4D))
    print(f"there are now {COUNT2} NANs")
    
    print("reassembling the case")
    for i in range (case_4D.shape[-1]):
        #remaking my dic of 3 D arrays channels[i] is the "DNB, M12" etc then fills with the array for the data
        filled[channels[i]] = new_case_4D[:,:,:,i] 
    return filled

def fillit2(case):
    channels = case['channels']
    COUNT1 = np.count_nonzero(np.isnan(case))
    print(f"there are {COUNT1} NANs")
    ##fill it
    new_case_4D = np.nan_to_num(case_4D, copy=True, nan=9999, posinf=None, neginf=None)
    ###count the NANs
    COUNT2 = np.count_nonzero(np.isnan(new_case_4D))
    print(f"there are now {COUNT2} NANs")
    return new_case_4D

def fillit_dic(case):
    filled = dict(fillit2(case[c], c) for c in channels)
    chans = list(case['channels']) + list(norms)
    new_case = {**case, **norms, 'channels': chans}
    return new_case




def NANfill(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    #fillcase = np.nan_to_num(case , copy=True, nan=9999, posinf=None, neginf=None)
    fillcase = fillit(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "_filledNAN.npz"
    np.savez_compressed(savepath,**fillcase )
    
       