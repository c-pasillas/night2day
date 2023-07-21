#takes the h5 for Mband and DNB + .nc for SM_Reflectance, colocates and saves as numpy array
# need to find a way to only do the ones we have all 3 files for ( otherwise it breaks mid process and have to deleate the files that arent a full set to move on)  can make this adjustment in the trio paired unpaired but then its hardcoded and only works for the VIIRS and the 3 files so could have to adjust if Mbands are seperated or for the ABI code
import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict
from satpy import Scene
from pyresample.geometry import SwathDefinition
import sys
import xarray
import crop
import common
from common import log, rgb, reset, blue, orange, bold


#lunar_data =['LunarZenithAngle', 'MoonIllumFraction' 'MoonPhaseAngle', 'LunarAzimuthAngle']
viirs_channels = ['DNB', 'M13', 'M14', 'M15', 'M16'] #M12 not needed
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']
SM_Reflectance = ['SM_Reflectance']
all_channels = viirs_channels + SM_Reflectance


def find_ncfile(trio):
    for f in trio:
        if f['filename'].endswith(".nc"):
            return f
def process_trio(trio, curr_idx, len_trio):
    """trio is a list of three parsed filenames (see the function parse_filename below).
    Given these three files, use Scene to load the appropriate channels.
    Then resample (colocate) to make the channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    dt = trio[0]["datetime"]
    log.info(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{len_trio}{reset} {blue}{dt}{reset}  ')
   
    #load the sat data
    scn = Scene(reader='viirs_sdr', filenames=[f['path'] for f in trio if f['filename'].endswith(".h5")])
    scn.load(viirs_channels + lat_long_both)
    #load and pair  the reflectance
    Reflectance = xarray.open_dataset(find_ncfile(trio)['path'])    
    swath_def = SwathDefinition(Reflectance['longitude'], Reflectance['latitude'])
    sm_refl = Reflectance['SM_Reflectance']
    sm_refl.attrs['area'] = swath_def
    #bring reflectance back to the satpy "Scene"
    scn['SM_Reflectance'] = sm_refl

    log.info(f'Resampling {blue}{dt}{reset}')
    resample_scn = scn.resample(scn['DNB'].attrs['area'], resampler='nearest')

    log.info(f'Cropping nan edges of {blue}{dt}{reset}')
    t = time.time()
    data = crop.crop_nan_edges_VIIRS(resample_scn, all_channels)
    log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    data['channels'] = list(data)
    data['filenames'] = [f['filename'] for f in trio]
    data["datetime"] = dt
    return data

# File name: GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b*
def parse_filename(path):
    """Given the path to one of the raw satellite sensor .h5 files and .nc reflectance, parse the
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
    #paired, unpaired = {}, {}
    #for dt, h5_list in d.items():
     #   p = paired if len(h5_list) == 3 else unpaired
      #  p[dt] = h5_list
    #return paired, unpaired
    return d

def grouped_h5s(h5_dir):
    h5s = [parse_filename(f) for f in h5_dir.iterdir() if f.suffix == '.h5' or f.suffix == '.nc']
    return group_by_datetime(h5s)

#### pick up here
def pack_case(args):
    """Scan for h5 and the nc files, match them by datetime, colocate and save them separately,
    then gather all samples together, crop to the minimum size, and save the entire
    case worth of channel data to a file, case.npz.
    This file, case.npz, contains each of the channels as separate array data.
    It also contains meta information like which channels are included and which h5 files
    went into making this case."""
    path = Path(args.h5_dir).resolve()
    log.info(f'h5_dir path is {path}')  
    trios = grouped_h5s(path)
    if len(trios) == 0:
        log.info(f"Error, couldn't find any paired .h5 files in {path}")
        sys.exit(-1)
  
    datas = [process_trio(trios[datetime], idx, len(trios)) for idx, datetime in enumerate(sorted(trios))]
    min_rows, min_cols = ft.reduce(crop.pairwise_min, [x['DNB'].shape for x in datas])
    channels = datas[0]['channels']
    case = {c: np.stack(tuple(d[c][:min_rows, :min_cols] for d in datas)) for c in channels}
    case['channels'] = channels

    d = case['DNB']
    d.clip(1e-11, out=d) # make erroneous sensor data non-negative
    np.multiply(d, 1e-4, out=d) # multiply by constant accounts for scaling factor in the Scene library
    
    
    #
    case['samples'] = [d["datetime"] for d in datas]
    filename = path / f'{path.name}_J01_RefALL_casefixed.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}')
    np.savez_compressed(filename, **case)
    log.info(f'Wrote {blue}{filename.name}{reset}')


