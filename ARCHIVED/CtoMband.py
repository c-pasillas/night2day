import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold


#dict of pairs
# M13 = C07
# M14 = C11
# M15 = C13
# M16 = C15


def C2M(case):
    channels = case['channels']
   
    new_case = {}
    for c in channels: 
        print(f"processing channel {c}")
        new_case[c]=case[c][:,:,start:end]   
        print(f'new dimensions are {new_case[c].shape}')
    arr_channels = sorted(list(channels)) 
    meta_channels = [ch for ch in case.files if ch not in arr_channels]
    for c in meta_channels:
        new_case[c] = case[c]
    return new_case


def main(args):
    case = np.load(args.npz_path)
    print(f"I loaded the file it contains {case.files}")  
    C2M_case  = C2M(case)       
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "C2M.npz"
    np.savez_compressed(savepath,**C2M_case)