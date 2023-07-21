#takes the h5 for Mband and DNB + .nc for SM_Reflectance, and .nc GOES file colocates and saves as numpy array

#end result is a dict of linked array  that has   DNB, SM_Reflectance ( my added variable), M13,M14,M15,M16, C07,C11,C13 and C15 colocated to the VIIRS DNB grid and only for the data form GOES that overlaps this.  looks like if I use the multiscence maybe i dont need my timestuff and it can do that for me and remove a bit of the precode.  This then would be run from my main "night2day"  as n2d ABI-raw-process DIRtoVIIRS  DIRtoGOES  but dont have it in there because its broke and i need the otehr stuff to run so a "code correct but not runable" is there as a place holder right now.  

#also need just a simple combo to make GOES only all in one .npz and brain keeps dumping that and not keeping location info somehow
​
import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
import datetime
from collections import defaultdict
from satpy import Scene
from pyresample.geometry import SwathDefinition
import sys
import xarray
import crop
#comment these out only used for printing and logging so will comment thouse out too
#import common
#from common import log, rgb, reset, blue, orange, bold
#import pprint
​
# TODO add any other interesting channels, eg shortwave
VIIRS_channels = ['DNB', 'M13', 'M14', 'M15', 'M16']
ABI_channels = ['C07', 'C11', 'C13','C15']
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']
SM_Reflectance = ['SM_Reflectance']
crop_channels = VIIRS_channels
​
all_channels = VIIRS_channels + ABI_channels + SM_Reflectance
​
format_dnb, format_abi = '%Y%m%d%H%M%S', '%Y%j%H%M%S'
​
​
def parse_time(time_str):
    return datetime.datetime.strptime(time_str, format_dnb if len(time_str) == 14 else format_abi)
​
​
ex_dnb_name = "GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b.h5"
def parse_filename_viirs(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    date = n[2][1:]
    start = parse_time(date + n[3][1:-1])
    end = parse_time(date + n[4][1:-1])
    return {'filename': name, 'channels': n[0], 'satellite': n[1],
            'start': start, 'end': end, 'path': str(path), "datetime": n[2] + '_' + n[3]}
​
ex_name = "OR_ABI-L1b-RadF-M3C15_G17_s20190781000382_e20190781011154_c20190781011198.docx"
def parse_filename_abi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    start = parse_time(n[3][1:-1])
    end = parse_time(n[4][1:-1])
    return {'filename': name, 'channels': n[0], 'satellite': n[2],
            'sat_and_start': f'{n[2]} {start}',
            'start': start, 'end': end, 'path': str(path)}
​
​
def gather_DNB(viirs_dir):
    return [parse_filename_viirs(f) for f in viirs_dir.iterdir() if f.suffix == '.h5' and 'DNB' in f.name]
​
​
def is_file_VIIRS(f): # answers if a file is VIIRS DNB or MBand
    return f.suffix ==".h5" or (f.suffix == ".nc" and "GDNBO" in f.name)
​
def group_by_datetime(files):
    """Given a list of parsed filenames match them up by
    corresponding datetime (so that the files can be colocated later by the
    Scene library. Filenames that don't have a paired file are returned separately."""
    def f(d, file):
        d[file['datetime']].append(file)
        return d
    d = ft.reduce(f, files, defaultdict(list))
    return d
​
def grouped_VIIRS(viirs_dir): #output is a list of DTG that has the DNB and Mband files and Reflectance
    VIIRS_time = [parse_filename_viirs(f) for f in viirs_dir.iterdir() if is_file_VIIRS(f)]
    return group_by_datetime(VIIRS_time)
​
def group_abi_by_time_sat(abi_dir):
    abis = [parse_filename_abi(f) for f in abi_dir.iterdir() if f.suffix == '.nc' and 'OR_ABI' in f.name]
    def red(d, abi):
        d[abi['sat_and_start']].append(abi)
        return d
    d = ft.reduce(red, abis, defaultdict(list))
    return d
​
def most_overlap(viirs_file, abi_lists):
    best_overlap, best_list = 0, []
    for abi_list in abi_lists:
        overlap = intersect(viirs_file, abi_list[0])
        if overlap > best_overlap:
            best_overlap, best_list = overlap, abi_list
    return best_list
​
def intersect(r1, r2):
    if r2['start'] < r1['start']:
        r1, r2 = r2, r1
    overlap = (r1['end'] - r2['start']).total_seconds()
    return 0 if overlap < 0 else overlap
​
def pair_viirs_with_abi(viirs_dict, abi_dict):
    abi_lists = list(abi_dict.values())
    ret = []
    for viirs_file in viirs_dict:
        best = most_overlap(viirs_file, abi_lists)
        ret.append((viirs_file, best))
    return ret #list of pairs of the VIIRS DTG and their associated ABI files
​
def match_stuff(VIIRS_pairing, ABI_DNB_pairing):
    ret = {}
    for dnb, abis in ABI_DNB_pairing:
        dt = dnb["datetime"]
        refl, dnb_mband = split_refl_and_viirs(VIIRS_pairing[dt])
        ret[dt] = {'reflectance': refl, 'viirs': dnb_mband, 'abi': abis}
    return ret
        
def split_refl_and_viirs(files):
    ref = None
    viirs = []
    for f in files:
        name = f['filename']
        if name.endswith(".nc") and name.startswith("GDNBO"):
            ref = f
        else:
            viirs.append(f)
    return ref, viirs       
        
def process_set(grouped_files, curr_idx, total_groups):
    """process_ALLis a list of parsed filenames (DNB, Mband, ABI, Cband)
    Given these files, use Scene to load the appropriate channels.
    Then resample (colocate) to make the channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    log.info(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{total_groups}{reset}')
    dt = grouped_files['viirs'][0]["datetime"]
    
    viirsfiles = [f["path"] for f in grouped_files['viirs']]
    abifiles =  [f["path"] for f in grouped_files['abi']]
    
    master_scene = Scene(filenames = {'viirs_sdr':viirsfiles, 'abi_l1b':abifiles})
    master_scene.load(VIIRS_channels + ABI_channels + lat_long_both)
  
    #load and pair  the reflectance
    reflectfile = grouped_files['reflectance']['path']
    Reflectance = xarray.open_dataset(reflectfile)    
    swath_def = SwathDefinition(Reflectance['longitude'], Reflectance['latitude'])
    sm_refl = Reflectance['SM_Reflectance']
    sm_refl.attrs['area'] = swath_def
    #bring reflectance back to the satpy "Scene"
    master_scene['SM_Reflectance'] = sm_refl
    resample_scn = master_scene.resample(master_scene['DNB'].attrs['area'], resampler='nearest')
    
    
    log.info(f'Cropping nan edges of {blue}{dt}{reset}')
    t = time.time()
    data = crop.crop_nan_edges(resample_scn, crop_channels, all_channels)
    log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')
​
    data['channels'] = list(data)
    data['filenames'] = viirsfiles + abifiles + [reflectfile]
    data["datetime"] = dt
    return data   
    
    
def main(args):
    """Scan for DNB, Mband, Reflectance and Cbands and get DTG to match up and processes colocation on."""
    viirs_dir, abi_dir = Path(args.viirs_dir).resolve(), Path(args.abi_dir).resolve()
    path = Path(args.viirs_dir).resolve()
    #master DTG is taken from what DNB files we have
    DNBs = gather_DNB(viirs_dir)
    #match up all the VIIRS DNB, Mband and Ref files
    VIIRS_pairing = grouped_VIIRS(viirs_dir)
    #match up the GOES ABI to the DNB
    ABI_dict = group_abi_by_time_sat(abi_dir)
    ABI_DNB_pairing = pair_viirs_with_abi(DNBs, ABI_dict)
      
    # make single file DTG list/dic of the DTGname and the associated DNB, Mband,Ref, and Cband so satpy/colocate etc gets called using this list of DTGs
    matched = match_stuff(VIIRS_pairing, ABI_DNB_pairing)
    #pprint.pprint(matched[list(matched)[0]])
    #process DTGs through the satpy colocation and save a final .npz of all the data
    #datas = [process_set(matched[datetime], idx, len(matched)) for idx, datetime in enumerate(sorted(matched)[0:45])]
    datas = [process_set(matched[datetime], idx, len(matched)) for idx, datetime in enumerate(sorted(matched))]
    min_rows, min_cols = ft.reduce(crop.pairwise_min, [x['DNB'].shape for x in datas])
    channels = datas[0]['channels']
    case = {c: np.stack(tuple(d[c][:min_rows, :min_cols] for d in datas)) for c in channels}
    case['channels'] = channels
​
    d = case['DNB']
    d.clip(1e-11, out=d) # make erroneous sensor data non-negative
    np.multiply(d, 1e-4, out=d) # multiply by constant accounts for scaling factor in the Scene library
    
    
    #
    case['samples'] = [d["datetime"] for d in datas]
    #filename2 = /gdata5/cpasilla/DATA/f'{path.name}_RefGOES_case.npz'
    filename = path / f'{path.name}_RefGOES_case.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}')
    np.savez_compressed(filename, **case)
    log.info(f'Wrote {blue}{filename.name}{reset}')
​