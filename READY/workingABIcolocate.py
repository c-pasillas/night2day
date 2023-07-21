import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict
from satpy import Scene
import datetime

import common
from common import log, rgb, reset, blue, orange, bold


#OR_ABI-L1b-RadF-M6C07_G17_s20201560830320_e20201560839398_c20201560839443.nc

all_channels = ['C07', 'C11', 'C13', 'C15']#, 'M15', 'M16']
lat_long = ['latitude', 'longitude']
format_abi = '%Y%j%H%M%S'
ex_name = "OR_ABI-L1b-RadF-M3C15_G17_s20190781000382_e20190781011154_c20190781011198.docx"

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

def process_pair(pair, out_path: Path, filename: Path):
    """Pair is a list of two parsed filenames (see the function parse_filename below).
    Given these two files, use Scene to load the appropriate channels.
    Then save these original channels (.png and optionally .nc files).
    Then resample (colocate) to make the sensor channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    log.info(f'Colocating {blue}{pair[0]["datetime"]}{reset}')
    scn = Scene(reader='abi_l1b', filenames=[f['path'] for f in pair])
    scn.load(all_channels + lat_long)
    save_datasets(scn, 'ORIGINAL_', str(out_path))

    log.info(f'Cropping nan edges of {blue}{pair[0]["datetime"]}{reset}')
    t = time.time()
    data = crop_nan_edges(save_datasets)
    log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    data['channels'] = list(data)
    data['filenames'] = [f['filename'] for f in pair]
    log.info(f'Saving {blue}{filename.name}{reset}')
    np.savez(filename, **data)



def processed_file(pair, out_path: Path, curr_idx, len_pairs):
    """Given a pair of parsed filenames, compute the colocated and NaN cropped array data.
    If this work has previously been done, we can just read the array data from the saved file."""
    f = out_path / (pair[0]['datetime'] + '.npz')
    log.debug(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{len_pairs}{reset}')
    if not f.exists():
        process_pair(pair, out_path, f)
    else:
        log.debug(f'Using previously computed {blue}{pair[0]["datetime"]}{reset}')
    return f

def parse_time(time_str):
    return datetime.datetime.strptime(time_str, format_abi)# if len(time_str) /= 14 else format_abi)

def parse_filename_abi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    start = parse_time(n[3][1:-1])
    end = parse_time(n[4][1:-1])
    return {'filename': name, 'channels': n[0], 'satellite': n[2],
            'sat_and_start': f'{n[2]} {start}',
            'start': start, 'end': end, 'path': str(path)}

def group_abi_by_time_sat(nc_dir):
    ncs = [parse_filename_abi(f) for f in nc_dir.iterdir() if f.suffix == '.nc']
    print ("ncs at zero is", ncs[0])
    def red(d, nc):
        d[nc['sat_and_start']].append(nc)
        return d
    d = ft.reduce(red, ncs, defaultdict(list))
    #print("this is D", d)
    return d


def group_by_datetime(ncs):
    """Given a list of parsed filenames of nc files, pair them up by
    corresponding datetime (so that these pairs can be colocated later by the
    Scene library. Filenames that don't have a paired file are returned separately."""
    def f(d, nc):
        d[nc['datetime']].append(nc)
        return d
    d = ft.reduce(f, ncs, defaultdict(list))
    paired, unpaired = {}, {}
    for dt, nc_list in d.items():
        p = paired if len(nc_list) == 4 else unpaired
        p[dt] = nc_list
    #print(paired)
    return paired, unpaired
    

def grouped_nc(nc_dir):
    ncs = [parse_filename(f) for f in nc_dir.iterdir() if f.suffix == '.nc']
    print("this is ncs" , ncs)
    return group_by_datetime(ncs)

def pair_with_ncs(nc_dict):
    nc_lists = list(nc_dict.values())
    ret = []
    best = nc_lists
    ret.append((best))
    print("thisis ret",ret)
    print("this is ret at ret[0]", ret[0])
    return ret

def pack_case(args):
    """Scan for nc files, pair them by datetime,
    then gather all samples together, crop to the minimum size, and save the entire
    case worth of channel data to a file, case.npz.
    This file, case.npz, contains each of the channels as separate array data.
    It also contains meta information like which channels are included and which h5 files
    went into making this case."""
    nc_dir =  Path(args.abi_dir).resolve()
    print("this is nc_dir", nc_dir)
    #h5s = gather_h5s(h5_dir)
    nc_dict = group_abi_by_time_sat(nc_dir)
    #print("this is nc dict", nc_dict)
    
    paired = pair_with_ncs(nc_dict)
    #print(paired)
    paired_sorted = (sorted(paired), key=lambda ncs: ncs[0]['start'])
    print("this is paried sorted" , paired_sorted)
    #for nc_list in paired_sorted:
        
     #   nc_list = int(sorted(nc_list, key=lambda nc: nc['start']))
     #   nc_first = nc_list[0]
        #print(f'nc_list first {nc_first}')
        #import sys
        #sys.exit()
      #  print(f'{bold}{nc_first["start"]} -> {nc_first["end"]}{reset}')
       # for nc in nc_list:
        #    print(f'{nc["filename"]}')
        #print()

    #print(paired_sorted)
    #files = [processed_file(paired[datetime], col, idx, len(paired)) for idx, datetime in enumerate(paired_sorted)]
    #npzs = [np.load(f) for f in files]
    #min_rows, min_cols = ft.reduce(pairwise_min, [x['C07'].shape for x in npzs])
    #channels = npzs[0]['channels']
    #case = {c: np.stack(tuple(npz[c][:min_rows, :min_cols] for npz in npzs)) for c in channels}
    #case['channels'] = channels
    
    #case['samples'] = [Path(f).stem for f in files]
    #filename = db_path.parent / 'case.npz'
    #log.info(f'Writing {blue}{filename.name}{reset}\n' +
     #        f'{orange}Channels{reset} {channels}\n{orange}Samples{reset} {case["samples"]}')
    #np.savez(filename, **case)
    #for npz in npzs:
     #   npz.close()
    #log.info(f'Wrote {blue}{filename.name}{reset}')



