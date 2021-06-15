import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

NSEW = 0, 0, 0, 0
zero_one = [0, -1]

def is_point_in_box(lat, long):
    n, s, e, w = NSEW
    if not s <= lat <= n:
        return False
    return (w <= long <= e) if w < e else (w <= long or long <= e)
def corners(lat_p, long_p):
    return [(lat_p[i], long_p[i]) for i in it.product(zero_one, zero_one)]
def patch_in_box(patch, lats, longs):
    cs = corners(lats[patch], longs[patch])
    if all(it.starmap(is_point_in_box, cs)):
        return True
    log.info(f'Rejecting {yellow}out-of-bounds{reset} patch {patch}, with corners {list(cs)}')
    return False
def filter_patches(patch_list, lats, longs):
    return [p for p in patch_list if patch_in_box(p, lats, longs)]

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

def aoi(args):
    path = Path(args.npz_path).resolve()
    global NSEW
    NSEW = args.NSEW
    if NSEW[0] == NSEW[1] == NSEW[2] == NSEW[3] == 0:
        NSEW = 45, -5, -105, 105
    log.info(f"Filtering images based on AOI {NSEW}")
    f = np.load(path)
    a_patches = all_patches(*f['DNB'].shape)
    aoi_patches = filter_patches(a_patches, f['latitude'], f['longitude'])
    log.info(f'Kept patches: {len(aoi_patches)} / {len(a_patches)}')

    g = {c: np.stack([f[c][p] for p in aoi_patches]) for c in f['channels']}
    g['channels'] = f['channels']
    out_path = path.parent / 'train_case.npz'
    log.info(f'Writing {blue}{out_path.name}{reset}')
    np.savez(out_path, **g)
    log.info(f'Wrote {blue}{out_path.name}{reset}')
