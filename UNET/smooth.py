import numpy as np
from scipy.signal import convolve

def smooth(amap,good,size):
    kern = np.ones(size)
    garr = np.zeros(amap.shape)
    garr[good] = 1
    aarr = np.zeros(amap.shape)
    aarr[good] = amap[good]
    smap = convolve(aarr,kern,mode='same')/convolve(garr,kern,mode='same')
    bad = ~np.isfinite(smap)
    smap[bad] = amap[bad]
    smap[~good] = amap[~good]
    return smap
