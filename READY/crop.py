import numpy as np
import itertools as it
import functools as ft
from satpy import Scene

all_channels = ['DNB', 'M12', 'M13', 'M14', 'M15', 'M16']
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']

def nan_count(row):
    ct = sum(1 for _ in it.takewhile(np.isnan, row))
    return ct if ct < len(row) / 2 else 0
def pairwise_max(n, m):
    return max(n[0], m[0]), max(n[1], m[1])
def pairwise_min(n, m):
    return min(n[0], m[0]), min(n[1], m[1])
def nan_edges(rows):
    """For each row, count the consecutive NaN blocks at the start and
    the end of the row. Then aggregate the max NaN count across all rows."""
    counts = ((nan_count(r), nan_count(r[::-1])) for r in rows)
    return ft.reduce(pairwise_max, counts)
def arrays_and_edges(scn: Scene):
    """Given a colocated scene, gather longitude and latitude arrays and
    sensor arrays. And return the maximum NaN edges found in each channel."""
    ae = [(arr, nan_edges(arr)) for arr in (scn[c].values for c in all_channels)]
    arrs = [scn[c].values for c in lat_long_both[:2]] + [a[0] for a in ae]
    return arrs, [a[1] for a in ae]
def crop_nan_edges(scn: Scene):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    arrs, edges = arrays_and_edges(scn)
    front, back = ft.reduce(pairwise_max, edges)
    till = arrs[0].shape[-1] - back
    return {name: arr[:, front:till] for name, arr in zip(lat_long + all_channels, arrs)}

    #log.info(f'Cropping nan edges of {blue}{dt}{reset}')
    #t = time.time()
    #data = crop_nan_edges(resample_scn)
    #log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

def flatten(lists):
    return [x for l in lists for x in l]

def combine_cases(cases):
    min_rows, min_cols = ft.reduce(pairwise_min, [case['latitude'].shape for case in cases])
    arr_channels = cases[0]['channels']
    meta_channels = [ch for ch in cases[0].files if ch not in arr_channels]
    ubercase = {c: np.stack(tuple(case[c][:min_rows, :min_cols] for case in cases))
                for c in arr_channels}
    metas = {c: flatten([list(case[c]) for case in cases])
             for c in meta_channels}
    comb = {**ubercase, **metas}
    return comb

