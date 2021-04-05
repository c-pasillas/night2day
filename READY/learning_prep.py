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
    """Given a 2-dimensional array of latitudes and longitudes, gather each of the
    four corners as (lat, long) pairs. That is, gather the indexes:
    (0, 0), (0, -1), (-1, 0), (-1, -1) which are the four corners of a 2-d array."""
    return [(lat_p[i], long_p[i]) for i in it.product(zero_one, zero_one)]
def patch_in_box(patch, lats, longs):
    """Given a patch identifier, something like (3, slice(256, 512), slice(256, 512))
    computes the latitude longitude of corners and checks if all corners are within
    the area-of-interest box."""
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
    """Given the number of samples and the dimensions of each sample,
    compute all patch identifiers. Each patch is a 2-dimensional array.
    Using 256 as patch_size, these are 256x256 image patches. Each patch identifier
    looks like (3, slice(256, 512), slice(256, 512)). This example patch identifier
    describes going to the fourth sample (index 3), and slicing out the patch of the
    overall 2-d array from 256-512 for both rows and columns.
    A patch identifier is useful because it can be directly used as an index into
    the arrays. If an array of DNB data is of shape (20, 3000, 4000), meaning 20 samples
    each of which is a 3000x4000 2-d array, we can access a patch of data by:
    DNB[p] where p is a patch identifier."""
    return list(it.product(range(samples), slices(x, patch_size), slices(y, patch_size)))

instructions = '# Lines starting with the pound sign (#) are not included\n' +\
               '# Mark the channel to predict by starting the line with an asterisk (*)\n'

def write_channel_options(path, channels):
    """Write out a description file displaying all the available channels that could be
    used as part of a learning algorithm."""
    with open(path, 'w') as f:
        f.write(instructions)
        for c in channels:
            f.write('  ' + c + '\n')

def read_channel_options(path):
    """Read and parse an edited file, which should indicate which channels to
    include in the learning process and which channel should be used as the
    predicted data."""
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
    """Read an .npz file to gather up the available sensor channel names."""
    with np.load(path) as f:
        chans = list(f['channels'])
        return [c for c in chans if c not in ('latitude', 'longitude')]

def shuffle_train_test(patch_list):
    """Given a list of patch identifiers, shuffle them and split them into
    train and test sets. Uses a seeded pseudo-random number generator, so
    the shuffle is repeatable for now. Uses train_proportion to determine
    how many patches should be used for training and testing."""
    random.seed('night2day')
    random.shuffle(patch_list)
    i = int(len(patch_list) * train_proportion)
    return patch_list[:i], patch_list[i:]

def stack_patches(patches, lats, longs, target, predictors):
    """Given patch identifiers, latitude and longitude arrays,
    the target channel (predicted) array, and the various predictor
    arrays, gather patch array data. The patch identifiers are coordinates
    used to index into the various channel arrays, and slice out patches of
    array data from them.
    If, say, 40 patch identifiers came in, and patch_size is 256, then the returned
    latitude, longitude and Y arrays would have shape (40, 256, 256).
    If predictors contained 7 arrays, the returned X array
    would have shape (40, 256, 256, 7). This is 40 patches, each 256x256 pixels,
    each pixels containing 7 channels of data."""
    t = time.time()
    lt, lg = np.stack([lats[p] for p in patches]), np.stack([longs[p] for p in patches])
    y = np.stack([target[p] for p in patches])
    x = np.stack([np.stack([pred[p] for pred in predictors], axis=-1) for p in patches])
    log.debug(f'Gathering patches took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')
    return {'Lat': lt, 'Lon': lg, 'Xdata': x, 'Ydata': y}

def suffix_keys(d, name_suffix):
    """Rename dictionary keys by adding a suffix to each key name."""
    return {k + name_suffix: v for k, v in d.items()}

def as_array(patches):
    """Given patch identifiers, convert each to a tuple of 5 numbers, so
    that they can be stored in a numpy array and saved for later."""
    return np.array([(sample, x.start, x.stop, y.start, y.stop) for sample, x, y in patches])

def ensure_ml_input(root_path: Path):
    ml_in = root_path / 'ML_INPUT'
    ml_in.mkdir(exist_ok=True)
    return ml_in

def make_unique_dir(ml_in: Path, predicted, predictors):
    """Given the channels of interest (predicted and predictors),
    make a unique folder to save the learning input files.
    If this combination of predicted and len(predictors) has been
    used before, try with '-1' '-2' '-3' etc until a unique
    folder name is found."""
    name = f'{predicted}-{len(predictors)}_predictors'
    x = ml_in / name
    if not x.exists():
        x.mkdir()
        return x
    for i in range(20):
        x = ml_in / (name + "-" + str(i))
        if not x.exists():
            x.mkdir()
            return x
    raise Exception("More than 20 folders with the same predicted channel, delete some")

def write_description(out_dir: Path, preds):
    with open(out_dir / 'description.txt', 'w') as d:
        d.write(f'Predicted channel: {preds["predicted"]}\n')
        d.write(f'Predictors:\n')
        for p in preds['predictors']:
            d.write(f'  {p}\n')

def learning_prep(db_path: Path):
    """Write learning_channels.txt with a menu of available channels for predictors and
    predicted, if it does not exist. After the user edits that file, selecting which
    channels to use, it:
    * computes all patch identifiers
    * filters patches based on area-of-interest
    * splits patches into test and train
    * gathers train and test data for each patch for each selected channel
    * creates a unique folder for this learning input
    * saves the array data and a description file into that folder"""
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

        ml_in = ensure_ml_input(db_path.parent)
        out_dir = make_unique_dir(ml_in, preds['predicted'], preds['predictors'])
        log.info(f'Writing {blue}learn.npz{reset}')
        np.savez(out_dir / 'data.npz', **zz, samples=f['samples'],
                 train_patches=as_array(train_patches), test_patches=as_array(test_patches),
                 y_channel=np.array([preds['predicted']]), x_channels=np.array(preds['predictors']))
        log.info(f'Wrote {blue}learn.npz{reset}')
        write_description(out_dir, preds)


