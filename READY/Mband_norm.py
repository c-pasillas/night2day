import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold

m_bands = ['M12', 'M13', 'M14', 'M15', 'M16']
bounds = {'M12': [230, 353],
          'M13': [230, 634],
          'M14': [190, 336],
          'M15': [190, 343],
          'M16': [190, 340]}

def normalize_band(arr, band):
    """Given the array data and the name of the band,
    Apply the transformation to the array data with the
    constants appropriate for that band.
    Returns the name of the normalized band, and the normalized array data."""
    mn, mx = bounds[band]
    ret = (mx - arr) / (mx - mn)
    ret.clip(max=1, out=ret)
    return band + 'norm', ret

def Mband_norms(case):
    """Given a case with array data for each raw channel,
    produce a dictionary with each of the normalized channels."""
    norms = dict(normalize_band(case[band], band) for band in m_bands)
    chans = list(case['channels']) + list(norms)
    new_case = {**case, **norms, 'channels': chans}
    return new_case


def mband_case (case):
    Mband_norms(case)
    return new_case
    


def mband(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    Mband_norm = Mband_norms(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_Mbandnorm.npz"
    np.savez(savepath,**Mband_norm)