#import statements
import numpy as np
import gzip
import math
import DNB_norm
import pathlib
import aoi
import patch
import NAN
        
#############
def patchnaoi(args):  
    print("im in FNN train and my args are", args)
    case = np.load(args.npz_path)
    print("I loaded the case") #case is large 3K4K
    
    case = patch.patch(case) # patch the case
    case = NAN.NAN(case) # remove patches with 9999
    if args.aoi:
        case = aoi.aoi_case(case, args.aoi)
          
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_trainready.npz"
    np.savez(savepath,**case)