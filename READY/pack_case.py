import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict


import common
from common import log, rgb, reset, blue, orange, bold

# TODO add any other interesting channels, eg shortwave
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

def crop_nan_edges(scn):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    arrs, edges = arrays_and_edges(scn)
    front, back = ft.reduce(pairwise_max, edges)
    till = arrs[0].shape[-1] - back
    return {name: arr[:, front:till] for name, arr in zip(lat_long + all_channels, arrs)}

def count_nan(array):
    return np.sum(np.isnan(array))
    
def is_all_nan(array):
    return all(np.isnan(array))

def find_nan_row(array):
    for i,row in enumerate(array):
        if is_all_nan(row):
            for j in range (i+1,len(array)):
                if not is_all_nan(array[j]):
                    return(i-1, j)
    return(0,0)     
    
def fill_in_nan_row(array, start, stop):
    a=(stop-start)//2
    for i in range(start + 1, start + a):
        array[i,:]=array[start,:]
    for i in range(start + a, stop):
        array[i,:]=array[stop,:]
            
def fill_in_nan_array(case, channels):
    log.info(f'filling in nan arrays for {channels}')
    for channel in channels:
        log.info(f'filling in nan arrays for {channel}')
        images = case[channel]   
        for i,image in enumerate(images):
            log.info(f'filling in image {i+1} / {len(images)}')
            start,stop = find_nan_row(image)
            if start != 0:
                fill_in_nan_row(image, start,stop)
        images[np.isnan(images)] = np.nanmean(images)
    d=case['DNB']
    d.clip(1e-11, out=d)
    

# TODO report NaN in the samples

def processed_file(pair, out_path: Path, curr_idx, len_pairs):
    """Given a pair of parsed filenames, compute the colocated and NaN cropped array data.
    If this work has previously been done, we can just read the array data from the saved file."""
    f = out_path / (pair[0]['datetime'] + '.npz')
    log.debug(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{len_pairs}{reset}')
    if not f.exists():
        print("im broke")#process_pair(pair, out_path, f)
    else:
        log.debug(f'Using previously computed {blue}{pair[0]["datetime"]}{reset}')
    return f

# File name: GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b*
def parse_filename(path):
    """Given the path to one of the raw satellite sensor .h5 files, parse the
    file name into separate fields, for ease of use in later logic."""
    n = path.name.split('_')
    return {'filename': path.name, 'channels': n[0], 'satellite': n[1],
            'date': n[2], 'start': n[3], 'end': n[4],
            'datetime': n[2] + '_' + n[3], 'path': str(path)}

def group_by_datetime(h5s):
    """Given a list of parsed filenames of h5 files, pair them up by
    corresponding datetime (so that these pairs can be colocated later by the
    Scene library. Filenames that don't have a paired file are returned separately."""
    def f(d, h5):
        d[h5['datetime']].append(h5)
        return d
    d = ft.reduce(f, h5s, defaultdict(list))
    paired, unpaired = {}, {}
    for dt, h5_list in d.items():
        p = paired if len(h5_list) == 2 else unpaired
        p[dt] = h5_list
    return paired, unpaired

def h5_dir_name(db_path):
    conn = sqlite3.connect(db_path)
    dir_name = conn.execute('select value from Settings where name=?', ('h5_dir_name',)).fetchone()[0]
    conn.close()
    return db_path.parent / dir_name #"RAWDATA" #dir_name

def grouped_h5s(h5_dir):
    h5s = [parse_filename(f) for f in h5_dir.iterdir() if f.suffix == '.h5']
    return group_by_datetime(h5s)

def ensure_colocated(db_path):
    col = db_path.parent / 'COLOCATED'
    col.mkdir(exist_ok=True)
    return col

# TODO make main entry to find unpaired samples
# TODO version that takes path to folder directly
#      then have this main command & control method call that
def pack_case(db_path: Path):
    """Scan for h5 files, pair them by datetime, colocate and save them separately,
    then gather all samples together, crop to the minimum size, and save the entire
    case worth of channel data to a file, case.npz.
    This file, case.npz, contains each of the channels as separate array data.
    It also contains meta information like which channels are included and which h5 files
    went into making this case."""
    h5_dir = h5_dir_name(db_path)
    pairs, unpaired = grouped_h5s(h5_dir)
    if unpaired:
        log.info(f'{rgb(255,0,0)}Unpaired h5s{reset} {unpaired}')
    col = ensure_colocated(db_path)
    files = [processed_file(pairs[datetime], col, idx, len(pairs)) for idx, datetime in enumerate(sorted(pairs))]
    files = files[:5]
    npzs = [np.load(f) for f in files]
    min_rows, min_cols = ft.reduce(pairwise_min, [x['DNB'].shape for x in npzs])
    channels = npzs[0]['channels']
    case = {c: np.stack(tuple(npz[c][:min_rows, :min_cols] for npz in npzs)) for c in channels}
    fill_in_nan_array(case, channels)
    case['channels'] = channels
    case['samples'] = [Path(f).stem for f in files]
    filename = db_path.parent / 'casereduced.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}Samples{reset} {case["samples"]}')
    np.savez(filename, **case)
    #np.savez(filename[:5,:,:,:],**case)
    for npz in npzs:
        npz.close()
    log.info(f'Wrote {blue}{filename.name}{reset}')

