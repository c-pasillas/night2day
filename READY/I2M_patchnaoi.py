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
    print("im in I2M PNA train and my args are", args)
    case = np.load(args.npz_path)
    print("I loaded the case") #case is large 3K4K
    
    case = patch.patchcase(case)
    print("I patched the case")
    # patch the case
    #case = NAN.NANcase(case) # remove patches with NANs
    #print("I removed patches with NANs in them")
    if args.aoi:
        case = aoi.aoi_case(case, args.aoi)
        print( "I am done with AOI cropping")      
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_I2M_PNA.npz"
    np.savez_compressed(savepath,**case)
    print("I saved the case")
    
    