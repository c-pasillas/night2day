import numpy as np
import itertools as it
import functools as ft
from satpy import Scene

import common
from common import log, rgb, reset, blue, orange, bold

#all_channels = ['DNB', 'M13', 'M14', 'M15', 'M16']

VIIRS_channels = ['DNB', 'M13', 'M14', 'M15', 'M16']
ABI_channels = ['C07', 'C11', 'C13','C14','C15']
AHI_channels = ['B07', 'B11','B13','B14','B15']
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']
SM_Reflectance = ['SM_Reflectance']
crop_channels = VIIRS_channels
Vall_channels = VIIRS_channels + SM_Reflectance
all_channels = VIIRS_channels + ABI_channels + SM_Reflectance
#crop_channels = VIIRS_channels


def nan_count(row):
    #ct = sum(1 for _ in it.takewhile(np.isnan, row))
    #return ct if ct < len(row) / 2 else 0
    return 0
def pairwise_max(n, m):
    return max(n[0], m[0]), max(n[1], m[1])
def pairwise_min(n, m):
    return min(n[0], m[0]), min(n[1], m[1])

def nan_edges(rows):
    """For each row, count the consecutive NaN blocks at the start and
    the end of the row. Then aggregate the max NaN count across all rows."""
    counts = ((nan_count(r), nan_count(r[::-1])) for r in rows)
    return ft.reduce(pairwise_max, counts)

def arrays_and_edges(scn: Scene, channels_crop):
    """Given a colocated scene, gather longitude and latitude arrays and
    sensor arrays. And return the maximum NaN edges found in each channel."""
    return [nan_edges(arr) for arr in (scn[c].values for c in channels_crop)]

def crop_nan_edges(scn: Scene, channels_crop = None, channels_keep = None):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    if channels_crop is None:
        channels_crop = all_channels
     
    if channels_keep is None:
        channels_keep = all_channels
    edges = arrays_and_edges(scn, channels_crop)
    print("edges are", edges)
    front, back = ft.reduce(pairwise_max, edges)
    print("front and back are", front, back)
    arrs = [scn[c].values for c in (lat_long_both[:2] + list(channels_keep))]
    till = arrs[0].shape[-1] - back 
    return {name: arr[:, front:till] for name, arr in zip(lat_long + channels_keep, arrs)}

def crop_nan_edges_VIIRS(scn: Scene, channels_crop = None, channels_keep = None):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    if channels_crop is None:
        #channels_crop = all_channels
        channels_crop = Vall_channels
    if channels_keep is None:
        channels_keep = Vall_channels
    edges = arrays_and_edges(scn, channels_crop)
    print("edges are", edges)
    front, back = ft.reduce(pairwise_max, edges)
    print("front and back are", front, back)
    arrs = [scn[c].values for c in (lat_long_both[:2] + list(channels_keep))]
    till = arrs[0].shape[-1] - back 
    return {name: arr[:, front:till] for name, arr in zip(lat_long + channels_keep, arrs)}

def crop_nan_edges_GOES(scn: Scene, channels_crop = None, channels_keep = None):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    if channels_crop is None:
        #channels_crop = all_channels
        channels_crop = ABI_channels
        #channels_crop = VIIRS_channels
    if channels_keep is None:
        channels_keep = ABI_channels
        #channels_keep = VIIRS_channels
    edges = arrays_and_edges(scn, channels_crop)
    print("edges are", edges)
    front, back = ft.reduce(pairwise_max, edges)
    print("front and back are", front, back)
    arrs = [scn[c].values for c in list(channels_keep)]
    till = arrs[0].shape[-1] - back 
    return {name: arr[:, front:till] for name, arr in zip(channels_keep, arrs)}

def crop_nan_edges_AHI(scn: Scene, channels_crop = None, channels_keep = None):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    if channels_crop is None:
        #channels_crop = all_channels
        channels_crop = AHI_channels
        #channels_crop = VIIRS_channels
    if channels_keep is None:
        channels_keep = AHI_channels
        #channels_keep = VIIRS_channels
    edges = arrays_and_edges(scn, channels_crop)
    print("edges are", edges)
    front, back = ft.reduce(pairwise_max, edges)
    print("front and back are", front, back)
    arrs = [scn[c].values for c in list(channels_keep)]
    till = arrs[0].shape[-1] - back 
    return {name: arr[:, front:till] for name, arr in zip(channels_keep, arrs)}

