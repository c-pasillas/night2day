#this makes sure you have the  norm channels and BTDs to predict on. does not cut the NADIR or the NANrows out.
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
    
    
def predict_all (case):
   
    #log.info("I'm removing the NANrows on the case")
    #case = NANrows.NANcase(case)
    #print("I finished NAN rows and have only kepts those with no NAN rows on the case")
    m_bands = ['M12norm', 'M13norm', 'M14norm', 'M15norm', 'M16norm', 'normBTD_M13M14', 'normBTD_M13M15', 'normBTD_M13M16', 'normBTD_M14M15', 'normBTD_M14M16','normBTD_M15M16',]
    c_bands = ['C07norm', 'C11norm', 'C13norm', 'C14norm', 'C15norm',   'normBTD_C07C11', 'normBTD_C07C13', 'normBTD_C07C14', 'normBTD_C07C15', 'normBTD_C11C13', 'normBTD_C11C14', 'normBTD_C11C15', 'normBTD_C13C15', 'normBTD_C13C14']
    
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
    
    Mcase = case
    for b in c_bands:
        Mcase.pop(b, None)                  
    Mcase['channels']= [c for c in Mcase if c not in['channels', 'samples']]
    
    Ccase = case
    for b in m_bands:
        Ccase.pop(b,None)                  
    Ccase['channels'] = [c for c in Ccase if c not in ['channels', 'samples']]
    
    return case, Mcase, Ccase
    
    #############
def main(args):
    case = np.load(args.npz_path)
    log.info(f"I loaded the case and arguments are {args.npz_path}") 
    case = predict_all(case)
    Mcase = predict_all(case)
    Ccase = predict_all(case)
    log.info(f"I am now saving case with channels {list(case)}, {list(Mcase)}, {list(Ccase)}.")
    #log.info(f"channels is {case['channels']} and samples is {case['samples']}")
    savepath = args.npz_path[:-4]+ f"_predict_master.npz"
    #np.savez_compressed(savepath, **case)
    savepath2 = args.npz_path[:-4]+ f"_predict_master_Mband.npz"
    np.savez_compressed(savepath2, **Mcase)
    savepath3 = args.npz_path[:-4]+ f"_predict_master_Cband.npz"
    np.savez_compressed(savepath3, **Ccase)
    log.info(f"I saved the cases {savepath}, {savepath2}, {savepath3}")
    
