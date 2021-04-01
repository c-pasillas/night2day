import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold

bounds = {'M12': [230, 353],
          'M13': [230, 634],
          'M14': [190, 336],
          'M15': [190, 343],
          'M16': [190, 340]}

def normalize_band(arr, band):
    mn, mx = bounds[band]
    ret = (mx - arr) / (mx - mn)
    ret.clip(max=1, out=ret)
    return band + 'norm', ret

def m_band_norms(case):
    return dict(normalize_band(case[band], band) for band in bounds)

def btd_and_norm(arr1, arr2, band1, band2):
    mn1, mx1, mn2, mx2 = bounds[band1] + bounds[band2]
    mx, mn = mx1 - mn2, mn1 - mx2
    arr = arr1 - arr2
    ret = (mx - arr) / (mx - mn)
    ret.clip(max=1, out=ret)
    name = 'BTD' + band1[1:] + band2[1:]
    return {name: arr, name + 'norm': ret}

def all_btd_norms(case):
    pairs = list(it.product(('M12', 'M13', 'M14'), ('M15', 'M16'))) + [('M15', 'M16')]
    btds = [btd_and_norm(case[b1], case[b2], b1, b2) for b1, b2 in pairs]
    return {k: v for btd in btds for k, v in btd.items()}

DNB_bounds = {'night': [2e-10, 3e-7],
              'full_moon': [1.26e-10, 1e-7],  # curtis
              'new_moon': [2e-11, 1e-9],
              'Miller_full_moon': [-9.5, -8.0]}

def formula(arr, mn_mx):
    mn, mx = mn_mx
    ret = (arr - mn) / (mx - mn)
    ret.clip(0, 1, out=ret)
    return ret

def reverse_formula(arr, mn_mx):
    mn, mx = mn_mx
    return (arr * (mx - mn)) + mn

def dnb_derive(dnb_arr):
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
    log.info(f'Computing normalized {orange}M bands{reset}')
    m_norms = m_band_norms(case)
    log.info(f'Computing normalized {orange}BTDs{reset}')
    btd_norms = all_btd_norms(case)
    log.info(f'Computing normalized {orange}DNB derivatives{reset}')
    dnb_norms = dnb_derive(case['DNB'])
    ch = list(case['channels']) + list(m_norms) + list(btd_norms) + list(dnb_norms)
    case['channels'] = ch
    return {**case, **m_norms, **btd_norms, **dnb_norms}

def show_stats(norm):
    for name, arr in norm.items():
        if name not in ('samples', 'channels'):
            log.debug(f'{name} max {blue}{np.nanmax(arr):.5}{reset} min {yellow}{np.nanmin(arr):.5}{reset}')

def normalize(db_path: Path):
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


