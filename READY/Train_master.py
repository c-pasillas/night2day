#import statements
import numpy as np
import gzip
import math
import pathlib
from common import log

import NANrows
import NADIR_crop
import band_norm
import btd
from common import log   
    
    
def I2M_all (case):
    log.info("I'm running NADIR on the case")
    case = NADIR_crop.NADIR(case)
    print("I finished NADIR crop on the case")
                

    log.info("I'm removing the NANrows on the case")
    case = NANrows.NANcase(case)
    print("I finished NAN rows and have only kepts those with no NAN rows on the case")
    
    #if mbands 
    log.info("I'm creating normbands")
    case = band_norm.band_norms(case)
    log.info("Done creating normbands")
    
    log.info("I'm creating btd bands")
    case = btd.btd_case(case)
    log.info("Done creating btd bands")
    
    
    for b in band_norm.bounds:
        case.pop(b,None)
    case.pop("DNB")
    case['channels'] = [c for c in case if c not in ['channels', 'samples']]
    return case
    
    #############
def main(args):
    case = np.load(args.npz_path)
    log.info(f"I loaded the case and arguments are {args.npz_path}") 
    case = I2M_all(case)
    log.info(f"I am now saving case with channels {list(case)}.")
    #log.info(f"channels is {case['channels']} and samples is {case['samples']}")
    savepath = args.npz_path[:-4]+ f"_trng_master.npz"
    np.savez_compressed(savepath, **case)
    log.info(f"I saved the case {savepath}")
    
