#takes the .nc GOES file colocates bands and saves as numpy array so .predit can be run on then ( wont have a truth though cuz wont have DNB SM_Reflectance matches)  The most operationalized ABI

#THIS IS FOR ABI ONLY WHEN YOU DONT NEED TO VALIDATE AGAINST A DNB
import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
import datetime
from collections import defaultdict
from satpy import Scene
from pyresample.geometry import SwathDefinition
import sys
import xarray
import crop
#comment these out only used for printing and logging so will comment thouse out too
import common
from common import log, rgb, reset, blue, orange, bold
import pprint
import gc

#ABI_channels = ['C07', 'C11', 'C13', 'C14','C15']
AHI_channels = ['B07', 'B11', 'B13', 'B14','B15']
#ABI_channels = ['C07', 'C11', 'C13','C15']
lat_long = ['latitude', 'longitude']
#crop_channels = ABI_channels
#all_channels = ABI_channels 
crop_channels = AHI_channels
all_channels = AHI_channels 

#format_abi = '%Y%j%H%M%S'
format_ahi = '%Y%m%d%H%M'

def parse_time(time_str): 
    return (datetime.datetime.strptime(time_str, format_ahi))
        
#ex_name = "OR_ABI-L1b-RadF-M3C15_G17_s20190781000382_e20190781011154_c20190781011198.docx"
#ex_name = "HS_H08_20200511_1800_B07_FLDSK_R20_S0110.dat"
def parse_filename_ahi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    #startint = (n[3][1:-1])
    #print(n[2])
    #print(n[3])
    #print(n[2]+n[3])
    start = parse_time(n[2]+n[3])
    print("this is the AHI start", start)
    startfix = start.strftime('%Y%m%d%H%M%S')
    print("startfixtime is", startfix)
    #end = parse_time(n[4][1:-1])
    #end = (n[4][1:-1])
    #print("this is hte ABI end", end)
    
    #startint = (n[3][1:-1])
    #start = parse_time(n[3][1:-1])
    #print("this is the ABI start", start)
    #startfix = start.strftime('%Y%m%d%H%M%S')
    #print("startfixtime is", startfix)
    #end = parse_time(n[4][1:-1])
    #end = (n[4][1:-1])
    #print("this is hte ABI end", end)
    return {'filename': name, 'channels': n[0], 'satellite': n[2],
            'sat_and_start': f'{n[1]} {start}',
            'start': start, 'path': str(path), "datetime": startfix}

def group_by_datetime(files): 
    """Given a list of parsed filenames match them up by
    corresponding datetime (so that the files can be colocated later by the
    Scene library. Filenames that don't have a paired file are returned separately."""
    def f(d, file):
        d[file['datetime']].append(file)
        return d
    d = ft.reduce(f, files, defaultdict(list))
    return d

def group_ahi_by_time_sat(ahi_dir):
    ahis = [parse_filename_ahi(f) for f in ahi_dir.iterdir() if f.suffix == '.DAT' and 'HS_H08' in f.name]
    def red(d, ahi):
        d[ahi['datetime']].append(ahi)
        return d
    d = ft.reduce(red, ahis, defaultdict(list))
    print("the ahis keys are times are", d)
    return d



def process_set_AHI(grouped_files, curr_idx, total_groups):
    """process_set_AHI takes a list of parsed filenames (Cband)
    Given these files, use Scene to load the appropriate channels.
    Then resample (colocate) to make the channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    log.info(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{total_groups}{reset}')
    #print("the grouped files are", grouped_files)
    
    #########
    #abi_dt = grouped_files['abi'][0]["datetime"]
    ahi_dt = grouped_files[0]["datetime"]
    print("the AHI dt is", ahi_dt)
    
    ahifiles =  [f["path"] for f in grouped_files]
    print("*************************************************the ahi files are**************************", ahifiles)
 
    master_scene = Scene(filenames = {'ahi_hsd':ahifiles})
    master_scene.load(AHI_channels)
  
    print(master_scene.available_dataset_names())
    print("datasets are now loaded")
    
    log.info(f'Cropping nan edges of {blue}{ahi_dt}{reset}')
    t = time.time()
    data = crop.crop_nan_edges_AHI(master_scene, None, None)
    #data = (master_scene, all_channels, all_channels)
    log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    data['channels'] = list(data)
    data['filenames'] = ahifiles 
    data["datetime"] = ahi_dt
    return data   
    
    
def main(args):
    """Scan for AHI bands and get DTG to match up and processes colocation on."""
    ahi_dir = Path(args.ahi_dir).resolve()
    print(ahi_dir)
    AHIpath = Path(args.ahi_dir).resolve() 
    #master DTG is taken from what GOES files we have
    #match up the GOES ABI to the DNB
    AHI_dict = group_ahi_by_time_sat(ahi_dir)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>ahi_dict is>>>>>>>>>>>>>>", AHI_dict)
    
    # make single file DTG list/dic of the DTGname and the Cband so satpy/colocate etc gets called using this list of DTGs
    matched = AHI_dict
    MATCHED = sorted(matched)
   # print("*******matched is", matched)
    #print("^^^^^^^^^^^matched keys are", matched.keys())
    print("*************************this is MATCHED***************/n", MATCHED)
    keylist = []
    for key in MATCHED:
        keylist.append(key)
    print("the keylist is", keylist) 
    
    #would love to make this a better looper so i can adjust the # of timestamps permitted based on space and FD vs MESO vs CONUS etc   20FD  timestamps is 25G going into FNNpredict
    
    
    datas = []
    dcount = 0
    for datetime in sorted(matched):#[0:100]:
        log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
        #datas.append(process_set_ABI(matched[datetime], dcount, len(matched)))
        try:
            datas.append(process_set_AHI(matched[datetime], dcount, len(matched)))
            dcount += 1
                         
        except KeyError:
            log.info(f"skipped {dcount} as there was no matching data")            
            continue
    
        except IndexError:
            log.info(f"skipped {dcount} due to an incomplete dataset")
            continue
            
    channels = datas[0]['channels']
    print("the channels are", channels)
    print("starting the stacking")
    case1 = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
    case1['channels'] = channels
    case1['samples'] = [d["datetime"] for d in datas]
    case1['AHItimes'] = [d['datetime'] for d in datas]
    
    filename = AHIpath / f'{AHIpath.name}_SAMPLE.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}')
    np.savez_compressed(filename, **case1)
    log.info(f'Wrote {blue}{filename.name}{reset}')       
    del(case1, datas)
    gc.collect()
            
