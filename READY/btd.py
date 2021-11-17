import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold


m_bands = ['M12','M13', 'M14', 'M15', 'M16']
trio_mband = ['M13', 'M14', 'M15', 'M16']
c_bands = ['C07', 'C11', 'C13', 'C15']


bounds = {'M12': [230, 353],
          'M13': [230, 634],
          'M14': [190, 336],
          'M15': [190, 343],
          'M16': [190, 340],
          'C07': [230, 634],
          'C11': [190, 336],
          'C13': [190, 343],
          'C15': [190, 340]}

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
    name = 'BTD_' + band1 + band2
    return {'norm' + name: ret} #only returning BTD norms ( no need to keep the BTD regulars)

def split_pairs(btd):
    b = btd.replace('BTD', '')
    return 'M' + b[:2], 'M' + b[2:]

def all_btd_norms(case, onlythese = None):
    """For several possible pairs of channels, compute the btd differences.
    Gather up all btd channels in a dictionary to return."""
    present_mbands = [b for b in m_bands if b in case]
    present_cbands = [b for b in c_bands if b in case]
    mpairs = list(filter(lambda pair: pair[0] < pair[1],
                            it.product(present_mbands, present_mbands)))
    cpairs = list(filter(lambda pair: pair[0] < pair[1],
                            it.product(present_cbands, present_cbands)))
    pairs = mpairs + cpairs
    ##TODO use only these pairs
    
    btds = [btd_and_norm(case[b1], case[b2], b1, b2) for b1, b2 in pairs]
    btd_norms = {k: v for btd in btds for k, v in btd.items()}
    chans = list(case['channels']) + list(btd_norms)
    new_case = {**case, **btd_norms, 'channels': chans}
    return new_case

def btd_case (case):
    new_case = all_btd_norms(case)
    return new_case 

def btd(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    BTDed = all_btd_norms(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_BTD.npz"
    np.savez_compressed(savepath,**BTDed)