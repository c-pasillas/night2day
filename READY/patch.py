
import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

patch_size = 1000
def slices(lim, step):
    return [slice(a, a + step) for a in range(0, lim - step + 1, step)]
def all_patches(samples, x, y):
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

def patch(args):
    pass
