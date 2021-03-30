import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict
from satpy import Scene

import common
from common import log, rgb, reset, blue, orange

# TODO add any other interesting channels, eg shortwave
all_channels = ['DNB', 'M12', 'M13', 'M14', 'M15', 'M16']
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']

def save_datasets(scene: Scene, tag, folder, save_nc=False):
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
    counts = ((nan_count(r), nan_count(r[::-1])) for r in rows)
    return ft.reduce(pairwise_max, counts)
def arrays_and_edges(scn: Scene):
    ae = [(arr, nan_edges(arr)) for arr in (scn[c].values for c in all_channels)]
    arrs = [scn[c].values for c in lat_long_both[:2]] + [a[0] for a in ae]
    return arrs, [a[1] for a in ae]
def crop_nan_edges(scn: Scene):
    arrs, edges = arrays_and_edges(scn)
    front, back = ft.reduce(pairwise_max, edges)
    till = arrs[0].shape[-1] - back
    return {name: arr[:, front:till] for name, arr in zip(lat_long + all_channels, arrs)}

def process_pair(pair, out_path: Path, filename: Path):
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

# TODO report NaN in the samples

def processed_file(pair, out_path: Path):
    f = out_path / (pair[0]['datetime'] + '.npz')
    if not f.exists():
        process_pair(pair, out_path, f)
    else:
        log.debug(f'Using previously computed {blue}{pair[0]["datetime"]}{reset}')
    return f

# File name: GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b*
def parse_filename(path):
    n = path.name.split('_')
    return {'filename': path.name, 'channels': n[0], 'satellite': n[1],
            'date': n[2], 'start': n[3], 'end': n[4],
            'datetime': n[2] + '_' + n[3], 'path': str(path)}

def group_by_datetime(h5s):
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
    return db_path.parent / dir_name

def grouped_h5s(h5_dir):
    h5s = [parse_filename(f) for f in h5_dir.iterdir() if f.suffix == '.h5']
    return group_by_datetime(h5s)

def ensure_colocated(db_path):
    col = db_path.parent / 'colocated'
    col.mkdir(exist_ok=True)
    return col

# TODO make main entry to find unpaired samples
# TODO version that takes path to folder directly
#      then have this main command & control method call that
def pack_case(db_path: Path):
    h5_dir = h5_dir_name(db_path)
    pairs, unpaired = grouped_h5s(h5_dir)
    if unpaired:
        log.info(f'{rgb(255,0,0)}Unpaired h5s{reset} {unpaired}')
    col = ensure_colocated(db_path)
    files = [processed_file(pairs[datetime], col) for datetime in sorted(pairs)]
    npzs = [np.load(f) for f in files]
    min_rows, min_cols = ft.reduce(pairwise_min, [x['DNB'].shape for x in npzs])
    channels = npzs[0]['channels']
    case = {c: np.stack(tuple(npz[c][:min_rows, :min_cols] for npz in npzs)) for c in channels}
    case['channels'] = channels
    case['samples'] = [Path(f).stem for f in files]
    filename = db_path.parent / 'case.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}Samples{reset} {case["samples"]}')
    np.savez(filename, **case)
    for npz in npzs:
        npz.close()


