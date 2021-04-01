import numpy as np
from scipy.stats import pearsonr
import sys

################################################################

refthrs_default = np.arange(5,55,5)

################################################################

def get_refc_stats(goes,mrms,\
    refthrs=refthrs_default,\
    ):

# inputs, required:
#   goes = goes refc
#   mrms = mrms refc

# inputs, optional:
#   refthrs = refc thresholds to evaluate statistics

# outputs:
#   stats = dictionary of stats

    good = (goes > -999) & (mrms > -999)

    # note: remove sub-zero variability
    goes[goes<0] = 0.
    mrms[mrms<0] = 0.

    stats = {}
    stats['ref'] = refthrs
    stats['pod'] = []
    stats['far'] = []
    stats['csi'] = []
    stats['bias'] = []
    stats['nrad'] = []
    stats['nsat'] = []
    stats['mean(goes-mrms)'] = np.mean(goes[good]-mrms[good])
    stats['std(goes-mrms)'] = np.std(goes[good]-mrms[good])
    stats['rmsd'] = np.sqrt(np.mean( (goes[good]-mrms[good])**2 ))
    stats['rsq'] = pearsonr(goes[good],mrms[good])[0]**2

    for rthr in refthrs:

        hasrad = mrms > rthr
        nrad = np.sum(hasrad)

        hassat = goes > rthr
        nsat = np.sum(hassat)

        if nrad == 0:
            stats['pod'].append(np.nan)
            stats['far'].append(np.nan)
            stats['csi'].append(np.nan)
            stats['bias'].append(np.nan)
            stats['nrad'].append(nrad)
            stats['nsat'].append(nsat)
            continue

        nhit = np.sum(  hasrad &  hassat & good )
        nmis = np.sum(  hasrad & ~hassat & good )
        nfal = np.sum( ~hasrad &  hassat & good )
        #nrej = np.sum( ~hasrad & ~hassat & good )

        try:
            csi = float(nhit) / float(nhit + nmis + nfal)
        except ZeroDivisionError:
            csi = np.nan
        try:
            pod = float(nhit) / float(nhit + nmis)
        except ZeroDivisionError:
            pod = np.nan
        try:
            far = float(nfal) / float(nhit + nfal)  #FA ratio
        except ZeroDivisionError:
            far = np.nan
        try:
            bias = float(nhit + nfal) / float(nhit + nmis)
        except ZeroDivisionError:
            bias = np.nan

        stats['pod'].append(pod)
        stats['far'].append(far)
        stats['csi'].append(csi)
        stats['bias'].append(bias)
        stats['nrad'].append(nrad)
        stats['nsat'].append(nsat)

    return stats

