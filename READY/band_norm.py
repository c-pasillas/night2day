import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold

m_bands = ['M12', 'M13', 'M14', 'M15', 'M16']
c_bands = ['C07', 'C11', 'C13', 'C14', 'C15']
b_bands = ['B07', 'B11', 'B13', 'B14', 'B15']

bounds = {'M12': [230, 353],
          'M13': [230, 634],
          'M14': [190, 336],
          'M15': [190, 343],
          'M16': [190, 340],
          'C07': [230, 634],
          'C11': [190, 336],
          'C13': [190, 343],
          'C14': [190, 343],
          'C15': [190, 340],
          'B07': [230, 634],
          'B11': [190, 336],
          'B13': [190, 343],
          'B14': [190, 343],
          'B15': [190, 340]}

def normalize_band(arr, band):
    """Given the array data and the name of the band,
    Apply the transformation to the array data with the
    constants appropriate for that band.
    Returns the name of the normalized band, and the normalized array data."""
    mn, mx = bounds[band]
    ret = (mx - arr) / (mx - mn)
    ret.clip(max=1, out=ret)
    return band + 'norm', ret

def band_norms(case):
    """Given a case with array data for each raw channel,
    produce a dictionary with each of the normalized channels."""
    bands = [b for b in bounds if b in case]
    norms = dict(normalize_band(case[band], band) for band in bands)
    chans = list(case['channels']) + list(norms)
    new_case = {**case, **norms, 'channels': chans}
    return new_case

def mband_case (case):
    new_case = band_norms(case)
    return new_case

def main(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    band_norm = band_norms(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_bandnorm.npz"
    np.savez_compressed(savepath,**band_norm)