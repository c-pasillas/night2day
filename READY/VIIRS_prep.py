#import statements
import numpy as np
import gzip
import math
import pathlib
import btd
import band_norm
import DNB_norm    
    
    
    
#############
def prep(args):  
    print("I loaded the case")
    case = np.load(args.npz_path)
    print ("Starting BTD calcs")
    case = btd.btd_case(case)
    print("starting Mband Norm")
    case = band_norm.band_case(case)
    print("starting DNB norm")
    case = DNB_norm.DNBnorm_case (case)
    savepath = args.npz_path[:-4]+ "_FINAL_FULL.npz"
    np.savez(savepath,**case)