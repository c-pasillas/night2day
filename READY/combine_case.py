import common
from common import log, bold, reset, color, rgb, blue
import functools as ft
import numpy as np
import crop


def same_channels(cases):
    common= set()
    sets = []
    for c in cases:
        channels = set(c['channels'])
        sets.append(channels)
    
    commonsets = sets[0]
    for s in sets[1:]:
           commonsets = commonsets & s
    
    arr_channels = sorted(list(commonsets)) 
    meta_channels = [ch for ch in cases[0].files if ch not in cases[0]['channels']]
    print(f"array channels is {arr_channels} and meta channels are {meta_channels}")
    return arr_channels, meta_channels        

def flatten(lists):
    return [x for l in lists for x in l]

def combine_cases(cases):
    shapes = [case['latitude'].shape[1:] for case in cases]
    log.info(f'shapes are {shapes}')
    min_rows, min_cols = ft.reduce(crop.pairwise_min, shapes)
    arr_channels, meta_channels = same_channels(cases)
    #arr_channels = cases[0]['channels']
    #meta_channels = [ch for ch in cases[0].files if ch not in arr_channels]
    log.info(f'Packing cropped array data')
    ubercase = {c: np.vstack(tuple(case[c][:, :min_rows, :min_cols] for case in cases))
                for c in arr_channels}
    log.info(f'Packing meta data')
    metas = {c: flatten([list(case[c]) for case in cases])
             for c in meta_channels}
    metas['channels'] = cases[0]['channels']
    comb = {**ubercase, **metas}
    return comb


def main(args):
    log.info(f'Starting combine cases')
    output = combine_cases([np.load(p) for p in args.npz_path])
    log.info(f'Writing {blue}{args.outputname}.npz{reset}')
    np.savez_compressed(args.outputname, **output)
    log.info(f'Wrote {blue}{args.outputname}.npz{reset}')


