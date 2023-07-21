#this makes sure you have the  norm channels and BTDs for GOES. does not cut the NADIR or the NANrows out.
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
    

#def GOESonly(case):
    
 #   VIIRS = ['M13', 'M14', 'M15', 'M16']
    
  #  for b in VIIRS:
   #     case.pop(b,None)
    #case['channels'] = [c for c in case if c not in ['channels', 'samples']]
    #print(case['channels'])
    #return case    
    
def predict_all (case):
    
    log.info("I'm creating normbands")
    case = band_norm.band_norms(case)
    log.info("Done creating normbands")
    
    REMOVELIST = ['M13norm', 'M14norm', 'M15norm', 'M16norm']
    
    for b in REMOVELIST:
        case.pop(b,None)
    log.info("I removed the Mbands")    
    
    log.info("I'm creating btd bands")
    case = btd.btd_case(case)
    log.info("Done creating btd bands")
    
    
    for b in band_norm.bounds:
       case.pop(b,None)
    
    REMOVELIST2 = ['M13norm', 'M14norm', 'M15norm', 'M16norm', 'normBTD_M13M14', 'normBTD_M13M15', 'normBTD_M13M16', 'normBTD_M14M15', 'normBTD_M14M16', 'normBTD_M15M16']
    
    for b in REMOVELIST2:
        case.pop(b,None)
        
    #case.pop("DNB")
    case['channels'] = [c for c in case if c not in ['channels', 'samples']]
    return case
    
    #############
def main(args):
    case = np.load(args.npz_path)
    log.info(f"I loaded the case and arguments are {args.npz_path}") 
    #remove the Mbands from the file
    
    #case = predict_all(GOESonly(case))
    
    case = predict_all(case)
    log.info(f"I am now saving case with channels {list(case)}.")
    #log.info(f"channels is {case['channels']} and samples is {case['samples']}")
    savepath = args.npz_path[:-4]+ f"_GOESTRNGMASTER.npz"
    np.savez_compressed(savepath, **case)
    log.info(f"I saved the case {savepath}")
    
