#import statements
import numpy as np
import gzip
import math
#import DNB_norm
import pathlib
import aoi
import patch
#import NAN
import common
from common import log, bold, reset, color, rgb, blue
import NADIR_crop

#############
def patchnaoi(args): 
    print("im in I2M PNA train and my args are", args)
    case = np.load(args.npz_path)
    print("I loaded the case") #case is large 3K4K
    print(f"I loaded the file it contains {case.files}")  
    
    #log.info("I'm running NADIR on the case")
    #case = NADIR_crop.NADIR(case)
    #case2 = NADIR_crop.NADIR(case)
    #print("I finished NADIR crop on the case")
    
    print("i am starting patching")  
    case = patch.patchcase(case)
    print("I patched the case")
    # patch the case
    
    if args.aoi:
        case = aoi.aoi_case(case, args.aoi)
        print( "I am done with AOI cropping")      
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_NADIR_PATCH_AOI-GOESVAL.npz"
    np.savez_compressed(savepath,**case)
    print("I saved the case")
    
    