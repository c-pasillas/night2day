import common
from common import log, bold, reset, color, rgb, blue
import functools as ft
import numpy as np
import crop

def flatten(lists):
    return [x for l in lists for x in l]
def combine_cases(cases):
    shapes = [case['latitude'].shape[1:] for case in cases]
    log.info(f'shapes are {shapes}')
    min_rows, min_cols = ft.reduce(crop.pairwise_min, shapes)
    arr_channels = cases[0]['channels']
    meta_channels = [ch for ch in cases[0].files if ch not in arr_channels]
    log.info(f'Packing cropped array data')
    ubercase = {c: np.vstack(tuple(case[c][:, :min_rows, :min_cols] for case in cases))
                for c in arr_channels}
    log.info(f'Packing meta data')
    metas = {c: flatten([list(case[c]) for case in cases])
             for c in meta_channels}
    metas['channels'] = cases[0]['channels']
    comb = {**ubercase, **metas}
    return comb

