import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

def main(args):
    case = np.load(args.npz_path)
    print(f"I loaded the file it contains {case.files}")  
    for c in case.files:     
        print(f"{c}: {case[c].shape}") 
    #C= np.count_nonzero(np.isnan(case['XXX']))
    #print(f"there are {C} NANs in the file