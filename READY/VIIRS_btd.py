import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold


#currently does all Mband paris ( but not the reverese). eventually would like to give it what pairs we care only.

m_bands = ['M12', 'M13', 'M14', 'M15', 'M16']
bounds = {'M12': [230, 353],
          'M13': [230, 634],
          'M14': [190, 336],
          'M15': [190, 343],
          'M16': [190, 340]}

def btd_and_norm(arr1, arr2, band1, band2):
    """Given two channels (band name and array data for each),
    apply the transformation to the arrays using the appropriate constants.
    Return the difference between channels, and the normalized difference.
    The return is a dictionary with these two entries."""
    mn1, mx1, mn2, mx2 = bounds[band1] + bounds[band2]
    mx, mn = mx1 - mn2, mn1 - mx2
    arr = arr1 - arr2
    ret = (mx - arr) / (mx - mn)
    ret.clip(max=1, out=ret)
    name = 'BTD' + band1[1:] + band2[1:]
    return {name + 'norm': ret} #only returning BTD norms ( no need to keep the BTD regulars)

def all_btd_norms(case, channels=None):
    """For several possible pairs of channels, compute the btd differences.
    Gather up all btd channels in a dictionary to return."""
    if not channels:
        channels = [c for c in case['channels'] if c in m_bands]
    pairs = list(filter(lambda pair: pair[0] < pair[1],
                        it.product(channels, channels)))
    btds = [btd_and_norm(case[b1], case[b2], b1, b2) for b1, b2 in pairs]
    btd_norms = {k: v for btd in btds for k, v in btd.items()}
    chans = list(case['channels']) + list(btd_norms)
    new_case = {**case, **btd_norms, 'channels': chans}
    return new_case

def btd_case (case):
    all_btd_norms(case)
    return new_case
    
    

def btd(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    BTDed = all_btd_norms(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_BTD.npz"
    np.savez(savepath,**BTDed)