import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold

bounds = {#'C01':
          #'C02':
          #'C03':
          #'C04':
          #'C05':
          #'C06':
          #'C07':
          #'C08':
          #'C09':
          #'C10':
          #'C11':
          #'C12':
          #'C13':
          #'C14':
          #'C15':
          #'C16':[,],
          'M12': [230, 353],
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

def band_norms(case):
    """Given a case with array data for each raw channel,
    produce a dictionary with each of the normalized channels."""
    return dict(normalize_band(case[band], band) for band in bounds)

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
    return {name: arr, name + 'norm': ret}

def all_btd_norms(case):
    """For several possible pairs of channels, compute the btd differences.
    Gather up all btd channels in a dictionary to return."""
    c = ('M12', 'M13', 'M14', 'M15', 'M16')
    pairs = list(filter(lambda pair: pair[0] != pair[1], it.product(c, c)))
    pairs_old = list(it.product(('M12', 'M13', 'M14'), ('M15', 'M16'))) + [('M15', 'M16')]
    btds = [btd_and_norm(case[b1], case[b2], b1, b2) for b1, b2 in pairs]
    return {k: v for btd in btds for k, v in btd.items()}

DNB_bounds = {'night': [2e-10, 3e-7],
              'full_moon': [1.26e-10, 1e-7],  # curtis
              'new_moon': [2e-11, 1e-9],
              'Miller_full_moon': [-9.5, -8.0]}

DNB_consts = {
     'DNB_norm': DNB_bounds['night'],
     'DNB_full_moon_norm': DNB_bounds['full_moon'],
     'DNB_new_moon_norm': DNB_bounds['new_moon'],
     'DNB_log_norm': np.log10(DNB_bounds['night']),
     'DNB_log_full_moon_norm': np.log10(DNB_bounds['full_moon']),
     'DNB_log_new_moon_norm': np.log10(DNB_bounds['new_moon']),
     'DNB_log_Miller_full_moon': DNB_bounds['Miller_full_moon']
}

def formula(arr, mn_mx):
    """Apply the DNB normalization to array data and appropriate constants."""
    mn, mx = mn_mx
    ret = (arr - mn) / (mx - mn)
    ret.clip(0, 1, out=ret)
    return ret

def reverse_formula(arr, mn_mx):
    """Reverses the DNB normalization formula to produce the original array data.
    This is actually not quite true. See the formula function above, it uses a clip
    to force negative values up to 0, and values above 1 down to 1. This operation
    cannot be reversed, so it is not possible to truly reverse the formula."""
    mn, mx = mn_mx
    return (arr * (mx - mn)) + mn

def denormalize(channel_name, arr):
    return reverse_formula(arr, DNB_consts[channel_name])

def dnb_derive(dnb_arr):
    """Given the DNB original array data, compute the various derivative channels.
    Multiplying by the constant 1e-4 accounts for a scaling factor applied in the Scene library
    (the source of the array data). For each of several derivative variants, apply the
    DNB normalization formula to the adjusted array data, or the log of it, with appropriate constants."""
    adj = dnb_arr * 1e-4
    ladj = np.log10(adj)
    r = {'DNBfix': adj,
         'DNB_norm': formula(adj, DNB_bounds['night']),
         'DNB_full_moon_norm': formula(adj, DNB_bounds['full_moon']),
         'DNB_new_moon_norm': formula(adj, DNB_bounds['new_moon']),
         'DNB_log_norm': formula(ladj, np.log10(DNB_bounds['night'])),
         'DNB_log_full_moon_norm': formula(ladj, np.log10(DNB_bounds['full_moon'])),
         'DNB_log_new_moon_norm': formula(ladj, np.log10(DNB_bounds['new_moon'])),
         'DNB_log_Miller_full_moon': formula(ladj, DNB_bounds['Miller_full_moon'])}
    return r

def normalize_case(case):
    """Given a case with the original channels (DNB and M/C bands), compute the various
    normalized and derived channels. It computes and combines normalized M bands, BTDs between
    pairs of M/C bands, and DNB derivatives."""
    log.info(f'Computing normalized {orange}M bands{reset}')
    bandnorms = band_norms(case)
    log.info(f'Computing normalized {orange}BTDs{reset}')
    btd_norms = all_btd_norms(case)
    log.info(f'Computing normalized {orange}DNB derivatives{reset}')
    dnb_norms = dnb_derive(case['DNB'])
    ch = list(case['channels']) + list(bandnorms) + list(btd_norms) + list(dnb_norms)
    case['channels'] = ch
    return {**case, **bandnorms, **btd_norms, **dnb_norms}

def show_stats(norm):
    for name, arr in norm.items():
        if name not in ('samples', 'channels'):
            log.debug(f'{name} max {blue}{np.max(arr):.5}{reset} min {yellow}{np.min(arr):.5}{reset}')

def normalize(db_path: Path):
    """Load case.npz, calculate normalized channels,
    then save all channel data out to case_norm.npz."""
    case_file = db_path.parent / 'case.npz'
    log.info(f'Loading {blue}{case_file.name}{reset}')
    with np.load(case_file) as f:
        case = dict(f)
    norm = normalize_case(case)
    show_stats(norm)
    norm_file = db_path.parent / 'case_norm.npz'
    log.info(f'Writing {blue}{norm_file.name}{reset}')
    np.savez(norm_file, **norm)
    log.info(f'Wrote {blue}{norm_file.name}{reset}')


