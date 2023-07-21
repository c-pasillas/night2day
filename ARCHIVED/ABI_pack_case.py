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

# TODO add any other interesting channels, eg shortwave
all_channels = ['DNB', 'M12', 'M13', 'M14', 'M15', 'M16']
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']
SAVE_IMAGES = False

def save_datasets(scene: Scene, tag, folder, save_nc=False):
    if SAVE_IMAGES:
        scene.save_datasets(datasets=all_channels, base_dir=folder, writer='simple_image',
                        filename=tag + '{start_time:%Y%m%d_%H%M%S}_{name}.png')
    if save_nc:
        scene.save_datasets(datasets=all_channels, base_dir=folder, writer='cf',
                            filename=tag + '{start_time:%Y%m%d_%H%M%S}_{name}.nc')

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
    scn = Scene(reader='viirs_sdr', filenames=[f['path'] for f in pair])
    scn.load(all_channels + lat_long_both)
    save_datasets(scn, 'ORIGINAL_', str(out_path))

    log.info(f'Resampling {blue}{pair[0]["datetime"]}{reset}')
    resample_scn = scn.resample(scn['DNB'].attrs['area'], resampler='nearest')

    log.info(f'Saving images {blue}{pair[0]["datetime"]}{reset}')
    t = time.time()
    save_datasets(resample_scn, 'COLOCATED_', str(out_path))
    log.debug(f'Saving images took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    log.info(f'Cropping nan edges of {blue}{pair[0]["datetime"]}{reset}')
    t = time.time()
    data = crop_nan_edges(resample_scn)
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

def intersect(r1, r2):
    if r2['start'] < r1['start']:
        r1, r2 = r2, r1
    overlap = (r1['end'] - r2['start']).total_seconds()
    return 0 if overlap < 0 else overlap

format_dnb, format_abi = '%Y%m%d%H%M%S', '%Y%j%H%M%S'
def parse_time(time_str):
    return datetime.datetime.strptime(time_str, format_dnb if len(time_str) == 14 else format_abi)

ex_dnb_name = "GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b.h5"
def parse_filename_dnb(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    date = n[2][1:]
    start = parse_time(date + n[3][1:-1])
    end = parse_time(date + n[4][1:-1])
    return {'filename': name, 'channels': n[0], 'satellite': n[1],
            'start': start, 'end': end, 'path': str(path)}

ex_name = "OR_ABI-L1b-RadF-M3C15_G17_s20190781000382_e20190781011154_c20190781011198.docx"
def parse_filename_abi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    start = parse_time(n[3][1:-1])
    end = parse_time(n[4][1:-1])
    return {'filename': name, 'channels': n[0], 'satellite': n[2],
            'sat_and_start': f'{n[2]} {start}',
            'start': start, 'end': end, 'path': str(path)}

def gather_h5s(h5_dir):
    return [parse_filename_dnb(f) for f in h5_dir.iterdir() if f.suffix == '.h5' and 'DNB' in f.name]
def group_abi_by_time_sat(nc_dir):
    ncs = [parse_filename_abi(f) for f in nc_dir.iterdir() if f.suffix == '.nc']
    def red(d, nc):
        d[nc['sat_and_start']].append(nc)
        return d
    d = ft.reduce(red, ncs, defaultdict(list))
    return d

def most_overlap(h5, nc_lists):
    best_overlap, best_list = 0, []
    for nc_list in nc_lists:
        overlap = intersect(h5, nc_list[0])
        if overlap > best_overlap:
            best_overlap, best_list = overlap, nc_list
    return best_list

def pair_h5s_with_ncs(h5s, nc_dict):
    nc_lists = list(nc_dict.values())
    ret = []
    for h5 in h5s:
        best = most_overlap(h5, nc_lists)
        ret.append((h5, best))
    return ret

# TODO version that takes path to folder directly
#      then have this main command & control method call that
def pack_case(args):
    """Scan for h5 files, pair them by datetime, colocate and save them separately,
    then gather all samples together, crop to the minimum size, and save the entire
    case worth of channel data to a file, case.npz.
    This file, case.npz, contains each of the channels as separate array data.
    It also contains meta information like which channels are included and which h5 files
    went into making this case."""
    h5_dir, nc_dir = Path(args.h5_dir).resolve(), Path(args.nc_dir).resolve()
    h5s = gather_h5s(h5_dir)
    nc_dict = group_abi_by_time_sat(nc_dir)
    global SAVE_IMAGES
    if args.save_images:
        SAVE_IMAGES = True
    """
    eleven_count = 0
    keys_in_order = sorted(nc_dict.keys())
    for key in keys_in_order:
        nc_list = sorted([f['filename'] for f in nc_dict[key]])
        if len(nc_list) == 16:
            eleven_count += 1
        print(f'{bold}{key}{reset}')
        for nc in nc_list:
            print(f'{nc}')
        print()

    print(f'There were {eleven_count}/{len(nc_dict)} groups with 16 files grouped.')
    print()
    print(f'{bold}H5 file start times{reset} length is {len(h5s)}')
    h5_starts = sorted([f['start'] for f in h5s])
    #h5_starts = sorted(h5s, key=lambda h5: h5['start'])
    for h5 in h5_starts:
        print(f'{h5}')
    """

    paired = pair_h5s_with_ncs(h5s, nc_dict)
    print(paired)
    paired_sorted = sorted(paired, key=lambda h5_ncs: h5_ncs[0]['start'])
    for h5, nc_list in paired_sorted:
        print(f'{bold}{h5["start"]} -> {h5["end"]}{reset}')
        print(f'{h5["filename"]}')
        nc_list = sorted(nc_list, key=lambda nc: nc['start'])
        nc_first = nc_list[0]
        #print(f'nc_list first {nc_first}')
        #import sys
        #sys.exit()
        print(f'{bold}{nc_first["start"]} -> {nc_first["end"]}{reset}')
        for nc in nc_list:
            print(f'{nc["filename"]}')
        print()

    print(paired_sorted)
    files = [processed_file(paired[datetime], col, idx, len(paired)) for idx, datetime in enumerate(paired_sorted)]
    npzs = [np.load(f) for f in files]
    min_rows, min_cols = ft.reduce(pairwise_min, [x['DNB'].shape for x in npzs])
    channels = npzs[0]['channels']
    case = {c: np.stack(tuple(npz[c][:min_rows, :min_cols] for npz in npzs)) for c in channels}
    d=case['DNB']
    d.clip(1e-11, out=d)
    case['channels'] = channels
    
    case['samples'] = [Path(f).stem for f in files]
    filename = db_path.parent / 'case.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}Samples{reset} {case["samples"]}')
    np.savez(filename, **case)
    for npz in npzs:
        npz.close()
    log.info(f'Wrote {blue}{filename.name}{reset}')

