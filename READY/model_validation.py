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

def write_channel_options(path, channels):
    """Write out a description file displaying all the available channels that could be
    used as part of a learning algorithm."""
    with open(path, 'w') as f:
        f.write(instructions)
        for c in channels:
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
        predicted_channel = f[preds["predicted"]]
        channel_names = preds['predictors']  
        channel_arrays = [f[c] for c in channel_names]        
        predictors_channels = np.stack(channel_arrays, axis=-1)      
        model_val_dir = ensure_ml_val(db_path.parent)
        out_dir = make_unique_dir(model_val_dir, preds['predicted'], preds['predictors'])
        log.info(f'Writing {blue}learn.npz{reset}')
        np.savez(out_dir / 'data.npz', Y=predicted_channel,X = predictors_channels,
                 y_channel=np.array([preds['predicted']]), x_channels=np.array(preds['predictors']))
        log.info(f'Wrote {blue}data.npz{reset}')
        write_description(out_dir, preds)


