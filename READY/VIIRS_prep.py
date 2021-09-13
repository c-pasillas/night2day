#import statements
import numpy as np
import gzip
import math
import pathlib
import VIIRS_btd
import Mband_norm
import DNB_norm    
    
    
    
#############
def prep(args):  
    print("I loaded the case")
    case = np.load(args.npz_path)
    print ("Starting BTD calcs")
    case = VIIRS_btd.btd(case)
    print("starting Mband Norm")
    case = Mband_norm.mband(case)
    print("starting DNB norm")
    case = DNB_norm.DNBnorm (case)
    savepath = args.npz_path[:-4]+ "_FINAL_FULL.npz"
    np.savez(savepath,**case)