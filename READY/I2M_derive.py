#import statements
import numpy as np
import gzip
import math
import DNB_norm
import pathlib
import Mband_norm
import DNB_norm
import VIIRS_btd
from common import log
        
#############
def main(args):
    case = np.load(args.npz_path)
    raws = {k: case[k] for k in ['M12', 'M13', 'M14', 'M15', 'M16', 'DNB']}
 
    log.info("I loaded the case") 
    
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
    case['channels'].extend (list(raws))
    log.info(f"I am now saving case with channels {list(case)}.")
    #log.info(f"channels is {case['channels']} and samples is {case['samples']}")
    savepath = args.npz_path[:-4]+ f"_NBD.npz"
    np.savez(savepath,**raws, **case)
    log.info("I saved the case")
    