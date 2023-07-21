import numpy as np
import random
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, bold, reset, yellow, blue, orange, rgb

instructions = '# Lines starting with the pound sign (#) are not included\n' +\
               '# Mark the channel to predict by starting the line with an asterisk (*)\n'

patch_size = 256

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

def write_channel_options(path, channels):
    """Write out a description file displaying all the available channels that could be
    used as part of a learning algorithm."""
    with open(path, 'w') as f:
        f.write(instructions)
        for c in channels:
            if "norm" not in c or 'DNB' in c:
                f.write('# ' + c + '\n')
            else:
                f.write('  ' + c + '\n')

def read_channel_options(path):
    """Read and parse an edited file, which should indicate which channels to
    include in the model val process and which channel should be used as the
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

def ensure_ml_val(root_path: Path):
    ml_in = root_path / 'ML_VALIDATION'
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
           
            
            
            
def model_val(db_path: Path):
    """Write model_val_channels.txt with a menu of available channels for predictors and
    predicted, if it does not exist. After the user edits that file, selecting which
    channels to use, it:
    * creates a unique folder for this learning input
    * saves the array data and a description file into that folder"""
    path = db_path.parent / 'model_val_channels.txt'
    if not path.exists():
        chans = gather_channels(db_path.parent / 'case_norm.npz')
        write_channel_options(path, chans)
        log.info(f'{bold}{yellow}Please edit{reset} {blue}{path.name}{reset} then run again.')
        return
    preds = read_channel_options(path)
    with np.load(db_path.parent / 'case_norm.npz') as f:
        a_patches = all_patches(*f['DNB'].shape)
        #first_patch = a_patches[0]
        #x=f['DNB'][first_patch]
       
        #DNB TRUTH
        predicted_channel = f[preds["predicted"]] #full size array (5, 3K, 4K)
        predicted_patch = np.stack([predicted_channel[p] for p in a_patches]) #patched array (40, 256,256)
        log.info(f'shape of predicted patch is {predicted_patch.shape}')
        #PREDICTORS
        channel_names = preds['predictors']  
        channel_arrays = [f[c] for c in channel_names]        
        predictors_channels = np.stack(channel_arrays, axis=-1)  #full size array (5,3K,4K,12)
        predictors_patch = np.stack([predictors_channels[p] for p in a_patches]) #patched array (40,256,256,12)
        log.info(f'shape of predictors patch is {predictors_patch.shape}')
        
        model_val_dir = ensure_ml_val(db_path.parent)
        out_dir = make_unique_dir(model_val_dir, preds['predicted'], preds['predictors'])
        log.info(f'Writing {blue}data.npz{reset}')
        ##or predicted/predictor_channel if can use full size
        np.savez(out_dir / 'data.npz', Y=predicted_patch,X = predictors_patch, 
                         y_channel=np.array([preds['predicted']]), x_channels=np.array(preds['predictors'])) 
        log.info(f'Wrote {blue}data.npz{reset}')
        write_description(out_dir, preds)


