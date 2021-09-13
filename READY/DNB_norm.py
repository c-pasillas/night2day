import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold


DNB_bounds = {'night': [2e-10, 3e-7],
              'full_moon': [1.26e-10, 1e-7],  # curtis
              'new_moon': [2e-11, 1e-9],
              'Miller_full_moon': [-9.5, -8.0]}

DNB_consts = {
     'DNB_norm': DNB_bounds['night'],
     'DNB_FMN': DNB_bounds['full_moon'],
     'DNB_NMN': DNB_bounds['new_moon'],
     'DNB_log_norm': np.log10(DNB_bounds['night']),
     'DNB_log_FMN': np.log10(DNB_bounds['full_moon']),
     'DNB_log_NMN': np.log10(DNB_bounds['new_moon']),
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
    print(" I am denormalizing")
    x= reverse_formula(arr, DNB_consts[channel_name])
    if "log" in channel_name:
        return 10 ** x
    else:
        return x


def dnb_derive(dnb_arr):
    """Given the DNB original array data, compute the various derivative channels.
    For each of several derivative variants, apply the
    DNB normalization formula to the adjusted array data, or the log of it, with appropriate constants."""
    adj = dnb_arr
    ladj = np.log10(adj)
    r = {#'DNBfix': adj,
         #'DNB_norm': formula(adj, DNB_bounds['night']),
         'DNB_FMN': formula(adj, DNB_bounds['full_moon']),
         # 'DNB_NMN': formula(adj, DNB_bounds['new_moon']),
         #'DNB_log_norm': formula(ladj, np.log10(DNB_bounds['night'])),
         'DNB_log_FMN': formula(ladj, np.log10(DNB_bounds['full_moon'])),
         # 'DNB_log_NMN': formula(ladj, np.log10(DNB_bounds['new_moon'])),
         #'DNB_log_Miller_full_moon': formula(ladj, DNB_bounds['Miller_full_moon'])
    }
    return r
        
def DNB_norms(case):
    """Given a case with  DNB """
    norms = dnb_derive(case["DNB"])
    chans = list(case['channels']) + list(norms)
    new_case = {**case, **norms, 'channels': chans}
    return new_case


def DNBnorm_case (case):
    DNB_norms(case)
    return new_case
    

def DNBnorm(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    DNB_norm = DNB_norms(case)
    print("I am now saving case")
    savepath = args.npz_path[:-4]+ "_DNBnorm.npz"
    np.savez(savepath,**DNB_norm)        
        