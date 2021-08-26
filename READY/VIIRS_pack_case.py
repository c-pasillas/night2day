import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict
from satpy import Scene
import sys

import crop
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

def process_pair(pair, image_dir: Path, curr_idx, len_pairs):
    """Pair is a list of two parsed filenames (see the function parse_filename below).
    Given these two files, use Scene to load the appropriate channels.
    Then save these original channels (.png and optionally .nc files).
    Then resample (colocate) to make the sensor channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    log.info(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{len_pairs}{reset}')
    dt = pair[0]["datetime"]
    log.info(f'Colocating {blue}{dt}{reset}')
    scn = Scene(reader='viirs_sdr', filenames=[f['path'] for f in pair])
    scn.load(all_channels + lat_long_both)
    #save_datasets(scn, 'ORIGINAL_', str(image_dir))

    log.info(f'Resampling {blue}{dt}{reset}')
    resample_scn = scn.resample(scn['DNB'].attrs['area'], resampler='nearest')

    log.info(f'Saving images {blue}{dt}{reset}')
    t = time.time()
    save_datasets(resample_scn, 'COLOCATED_', str(image_dir))
    log.debug(f'Saving images took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    data = crop.crop_nan_edges(resample_scn)

    data['channels'] = list(data)
    data['filenames'] = [f['filename'] for f in pair]
    data["datetime"] = dt
    return data

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

def grouped_h5s(h5_dir):
    h5s = [parse_filename(f) for f in h5_dir.iterdir() if f.suffix == '.h5']
    return group_by_datetime(h5s)

def ensure_image_dir(path):
    col = path / 'IMAGES'
    col.mkdir(exist_ok=True)
    return col

def pack_case(args):
    """Scan for h5 files, pair them by datetime, colocate and save them separately,
    then gather all samples together, crop to the minimum size, and save the entire
    case worth of channel data to a file, case.npz.
    This file, case.npz, contains each of the channels as separate array data.
    It also contains meta information like which channels are included and which h5 files
    went into making this case."""
    path = Path(args.h5_dir).resolve()
    log.info(f'h5_dir path is {path}')
    global SAVE_IMAGES
    if args.save_images:
        SAVE_IMAGES = True
    pairs, unpaired = grouped_h5s(path)
    if len(pairs) == 0:
        log.info(f"Error, couldn't find any paired .h5 files in {path}")
        sys.exit(-1)
    if unpaired:
        log.info(f'{rgb(255,0,0)}Unpaired h5s{reset} {unpaired}')
    image_dir = ensure_image_dir(path)
    log.info(f'image_dir is {image_dir}')

    datas = [process_pair(pairs[datetime], image_dir, idx, len(pairs)) for idx, datetime in enumerate(sorted(pairs))]
    min_rows, min_cols = ft.reduce(crop.pairwise_min, [x['DNB'].shape for x in datas])
    channels = datas[0]['channels']
    case = {c: np.stack(tuple(d[c][:min_rows, :min_cols] for d in datas)) for c in channels}
    case['channels'] = channels

    d = case['DNB']
    d.clip(1e-11, out=d) # make erroneous sensor data non-negative
    np.multiply(d, 1e-4, out=d) # multiply by constant accounts for scaling factor in the Scene library

    case['samples'] = [d["datetime"] for d in datas]
    filename = path / f'{path.name}_case.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}')
    np.savez(filename, **case)
    log.info(f'Wrote {blue}{filename.name}{reset}')

