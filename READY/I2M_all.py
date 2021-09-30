#import statements
import numpy as np
import gzip
import math
import DNB_norm
import pathlib
import aoi
import patch
import NAN
import Mband_norm
import DNB_norm
import VIIRS_btd
from common import log
                


def I2M_all_case (case):
    new_case = I2M_all(case)
    return new_case    
    
    
def I2M_all (case,args):
    log.info("I'm patching the case")
    case = patch.patchcase(case)
    print("I patched the case")
                
    log.info("I'm cropping to the AIO")
    if args.aoi:
        case = aoi.aoi_case(case, args.aoi)
    print( "I am done with AOI cropping")
    
    
    log.info("I'm creating mbands")
    mbands = [itm.replace('norm', '') for itm in args.Predictors if itm.startswith('M')]
    if mbands:
        case = Mband_norm.Mband_norms(case, bands=mbands)
    log.info("Done creating mbands")
    
    log.info("I'm creating btd bands")
    btd_bands = [itm.replace('norm', '') for itm in args.Predictors if itm.startswith('BTD')]
    if btd_bands:
        case = VIIRS_btd.btd_case(case, derive=btd_bands)
    log.info("Done creating btd bands")

    log.info("Creating DNB bands")
    case = DNB_norm.DNBnorm_case(case)
    log.info("Done creating DNB bands")
    
    for b in Mband_norm.m_bands + ['DNB']:
        case.pop(b)
    case['channels'] = [c for c in case if c not in ['channels', 'samples']]
    return case
    
    #############
def main(args):
    case = np.load(args.npz_path)
    log.info(f"I loaded the case and arguments are {args.npz_path}, {args.aoi}, {args.Predictors}") 
    case = I2M_all(case, args)
    log.info(f"I am now saving case with channels {list(case)}.")
    #log.info(f"channels is {case['channels']} and samples is {case['samples']}")
    savepath = args.npz_path[:-4]+ f"_I2M_ALL.npz"
    np.savez_compressed(savepath, **case)
    log.info("I saved the case")
    