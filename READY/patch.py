
import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

def slices(lim, step):
    return [slice(a, a + step) for a in range(0, lim - step + 1, step)]
def all_patches(samples, x, y, patch_size):
    """Given the number of samples and the dimensions of each sample,
    compute all patch identifiers. Each patch is a 2-dimensional array.
    Using 256 as patch_size, these are 256x256 image patches. Each patch identifier
    looks like (3, slice(256, 512), slice(256, 512)). This example patch identifier
    describes going to the fourth sample (index 3), and slicing out the patch of the
    overall 2-d array from 256-512 for both rows and columns.
    A patch identifier is useful because it can be directly used as an index into
    the arrays. If an array of DNB data is of shape (20, 3000, 4000), meaning 20 samples
    each of which is a 3000x4000 2-d array, we can access a patch of data by:
    DNB[p] where p is a patch identifier."""
    return list(it.product(range(samples), slices(x, patch_size), slices(y, patch_size)))

def patch_case(case, patch_size=256):
    shape = case['latitude'].shape
    a_patches = all_patches(shape[0], shape[1], shape[2], patch_size)
    arr_channels = case['channels']
    meta_channels = [ch for ch in cases.files if ch not in arr_channels]
    log.info(f'Patching channels: {channels}')
    arr_data = {c: np.stack([case[c][p] for p in a_patches], axis=-1)
                for c in arr_channels}
    metas = {c: case[c] for c in meta_channels}
    patch_samples = [case['samples'][p[0]] for p in a_patches]
    metas['samples'] = patch_samples
    new_case = {**arr_data, **metas}
    return new_case

def patch(args):
    pass
