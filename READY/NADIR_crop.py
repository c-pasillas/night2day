import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold


def NADIR(case):
    channels = case['channels']
    width = case[channels[0]].shape[-1]
    center = int(width/2)
    start = int(center - 600)
    end = int(center + 600)
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
    NADIR_case  = NADIR(case)       
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "NADIR.npz"
    np.savez_compressed(savepath,**NADIR_case)

