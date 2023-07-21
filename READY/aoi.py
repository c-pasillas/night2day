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

def aoi_case(case,nsew):
    global NSEW
    NSEW = nsew
    if NSEW[0] == NSEW[1] == NSEW[2] == NSEW[3] == 0:
        NSEW = 50, 25, -130, 180
    log.info(f"Filtering images based on AOI {NSEW}")
    a_patches = range(len(case['latitude']))
    aoi_patches = filter_patches(a_patches, case['latitude'], case['longitude'])
    log.info(f'Kept patches: {len(aoi_patches)} / {len(a_patches)}')
    channels_to_filter = list(case['channels']) + ['samples']
    log.info(f'starting to stack the good patches')
    
    #in_aoi_case = {c: case[c][aoi_patches] for c in channels_to_filter}
    in_aoi_case = {}
    for c in channels_to_filter:
        print("starting channel", c)
        x = case[c][aoi_patches]
        in_aoi_case[c] =x
    
    in_aoi_case['channels'] = case['channels']
    in_aoi_case['aoi'] = [f'{NSEW}']
    log.info(f'done stacking the patches')
    return in_aoi_case

def aoi_by_pixel(case, nsew): #goes pixel by pixel and only saves the pixels in the AOI
    global NSEW
    NSEW = nsew
    if NSEW[0] == NSEW[1] == NSEW[2] == NSEW[3] == 0:
        NSEW = 50, 25, -130, 180
    log.info(f"Filtering pixels based on AOI {NSEW}")
    lat= case['latitude'].flatten()
    long = case['longitude'].flatten()
    channels_to_filter = list(case['channels'])
    keep_points = [idx for idx in range(len(lat)) if is_point_in_box(lat[idx],long[idx])]
    pixels_in_aoi = {}
    for c in channels_to_filter:
        print("starting channel", c)
        x = case[c].flatten()[keep_points]
        pixels_in_aoi[c] =x
    pixels_in_aoi['channels'] = case['channels']
    pixels_in_aoi['aoi'] = [f'{NSEW}']
    log.info(f'done stacking the pixel channels')
    
    #flat_reflect = case['SM_reflectance'].flatten()
    #img_array = np.zeros(len(flat_reflect))
    #img_array[keep_points] = flat_reflect[keep_points]
    #img_array = img_array.reshape(case['SM_reflectance'].shape)
    
    return pixels_in_aoi#,img_array
    
def aoi(args):
    path = Path(args.npz_path).resolve()
    f = np.load(path)
    print(f)
    if args.pixel:
        g = aoi_by_pixel(f, args.NSEW)
        savepath = args.npz_path[:-4]+ f"_aoi_pixel_{NSEW[0]}_{NSEW[1]}_{NSEW[2]}_{NSEW[3]}.npz" 
    else:
        g = aoi_case(f, args.NSEW)
        savepath = args.npz_path[:-4]+ f"_aoi_{NSEW[0]}_{NSEW[1]}_{NSEW[2]}_{NSEW[3]}.npz"
    np.savez_compressed(savepath, **g)
    log.info(f'Wrote {blue}{savepath}{reset}')

