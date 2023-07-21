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
    print(arr_channels)
    meta_channels = [ch for ch in case.files if ch not in arr_channels]
    print(meta_channels)
    for c in meta_channels:
        new_case[c] = case[c]
    new_case = {**new_case,'channels': channels}   
    #metas = {c: case[c] for c in meta_channels}    
    #new_case = {**case, **norms, 'channels': chans}
    #new_case = {**new_case,'channels': channels}
    #new_case = {**new_case,**metas}    
    return new_case


def main(args):
    case = np.load(args.npz_path)
    print(f"I loaded the file it contains {case.files}")  
    NADIR_case  = NADIR(case)       
    print("I am now saving case")
    savepath = args.npz_path[:-4] + "_NADIR.npz"
    np.savez_compressed(savepath,**NADIR_case)

