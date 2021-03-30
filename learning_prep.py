import numpy as np
import random
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, bold, reset, yellow, blue, orange, rgb

# TODO change area-of-interest here
NSEW = 45, -5, -125, 160
patch_size = 256
train_proportion = 0.8
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
    log.debug(f'Rejecting {yellow}out-of-bounds{reset} patch {patch}, with corners {list(cs)}')
    return False
def filter_patches(patch_list, lats, longs):
    return [p for p in patch_list if patch_in_box(p, lats, longs)]

def slices(lim, step):
    return [slice(a, a + step) for a in range(0, lim - step + 1, step)]
def all_patches(samples, x, y):
    return list(it.product(range(samples), slices(x, patch_size), slices(y, patch_size)))

instructions = '# Lines starting with the pound sign (#) are not included\n' +\
               '# Mark the channel to predict by starting the line with an asterisk (*)\n'

def write_channel_options(path, channels):
    with open(path, 'w') as f:
        f.write(instructions)
        for c in channels:
            f.write('  ' + c + '\n')

def read_channel_options(path):
    with open(path) as f:
        lines = [line.strip() for line in f.readlines()]
    lines = [line for line in lines if line and not line.startswith('#')]
    asterisks = [line[1:].strip() for line in lines if line.startswith('*')]
    if not asterisks:
        asterisks = lines[:1]
        channels = lines[1:]
    else:
        channels = [c for c in lines if not c.startswith('*')]
    return {'predicted': asterisks[0], 'predictors': channels}

def gather_channels(path):
    with np.load(path) as f:
        chans = list(f['channels'])
        return [c for c in chans if c not in ('latitude', 'longitude')]

def shuffle_train_test(patch_list):
    random.seed('night2day')
    random.shuffle(patch_list)
    i = int(len(patch_list) * train_proportion)
    return patch_list[:i], patch_list[i:]

def stack_patches(patches, lats, longs, target, predictors):
    t = time.time()
    lt, lg = np.stack([lats[p] for p in patches]), np.stack([longs[p] for p in patches])
    y = np.stack([target[p] for p in patches])
    x = np.stack([np.stack([pred[p] for pred in predictors], axis=-1) for p in patches])
    log.debug(f'Gathering patches took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')
    return {'Lat': lt, 'Lon': lg, 'Xdata': x, 'Ydata': y}

def suffix_keys(d, name_suffix):
    return {k + name_suffix: v for k, v in d.items()}

def as_array(patches):
    return np.array([(sample, x.start, x.stop, y.start, y.stop) for sample, x, y in patches])

def learning_prep(db_path: Path):
    path = db_path.parent / 'learning_channels.txt'
    if not path.exists():
        chans = gather_channels(db_path.parent / 'case_norm.npz')
        write_channel_options(path, chans)
        log.info(f'{bold}{yellow}Please edit{reset} {blue}{path.name}{reset} then run again.')
        return
    preds = read_channel_options(path)
    with np.load(db_path.parent / 'case_norm.npz') as f:
        a_patches = all_patches(*f['DNB'].shape)
        log.info(f'Filtering patches based on area-of-interest (AOI)')
        aoi_patches = filter_patches(a_patches, f['latitude'], f['longitude'])
        train_patches, test_patches = shuffle_train_test(aoi_patches)
        lats, longs, target = f['latitude'], f['longitude'], f[preds['predicted']]
        predictors = [f[c] for c in preds['predictors']]
        log.info(f'Gathering train patches')
        train_z = stack_patches(train_patches, lats, longs, target, predictors)
        log.info(f'Gathering test patches')
        test_z = stack_patches(test_patches, lats, longs, target, predictors)
        zz = {**suffix_keys(train_z, '_train'), **suffix_keys(test_z, '_test')}
        log.info(f'Writing {blue}learn.npz{reset}')
        np.savez(db_path.parent / 'learn.npz', **zz, samples=f['samples'],
                 train_patches=as_array(train_patches), test_patches=as_array(test_patches),
                 y_channel=np.array([preds['predicted']]), x_channels=np.array(preds['predictors']))


