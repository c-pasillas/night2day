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

ABI_channels = ['C07', 'C11', 'C13', 'C14','C15']
#ABI_channels = ['C07', 'C11', 'C13','C15']
lat_long = ['latitude', 'longitude']
crop_channels = ABI_channels
all_channels = ABI_channels 


format_abi = '%Y%j%H%M%S'

def parse_time(time_str): 
    return (datetime.datetime.strptime(time_str, format_abi))
        
#ex_name = "OR_ABI-L1b-RadF-M3C15_G17_s20190781000382_e20190781011154_c20190781011198.docx"
def parse_filename_abi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    #startint = (n[3][1:-1])
    start = parse_time(n[3][1:-1])
    #print("this is the ABI start", start)
    startfix = start.strftime('%Y%m%d%H%M%S')
    #print("startfixtime is", startfix)
    end = parse_time(n[4][1:-1])
    #end = (n[4][1:-1])
    #print("this is hte ABI end", end)
    return {'filename': name, 'channels': n[0], 'satellite': n[2],
            'sat_and_start': f'{n[2]} {start}',
            'start': start, 'end': end, 'path': str(path), "datetime": startfix}

def group_by_datetime(files): 
    """Given a list of parsed filenames match them up by
    corresponding datetime (so that the files can be colocated later by the
    Scene library. Filenames that don't have a paired file are returned separately."""
    def f(d, file):
        d[file['datetime']].append(file)
        return d
    d = ft.reduce(f, files, defaultdict(list))
    return d

def group_abi_by_time_sat(abi_dir):
    abis = [parse_filename_abi(f) for f in abi_dir.iterdir() if f.suffix == '.nc' and 'OR_ABI' in f.name]
    def red(d, abi):
        d[abi['datetime']].append(abi)
        return d
    d = ft.reduce(red, abis, defaultdict(list))
    #print("the abis keys are times are", d)
    return d



def process_set_ABI(grouped_files, curr_idx, total_groups):
    """process_set_ABI takes a list of parsed filenames (Cband)
    Given these files, use Scene to load the appropriate channels.
    Then resample (colocate) to make the channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    log.info(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{total_groups}{reset}')
    #print("the grouped files are", grouped_files)
    
    #########
    #abi_dt = grouped_files['abi'][0]["datetime"]
    abi_dt = grouped_files[0]["datetime"]
    print("the ABI dt is", abi_dt)
    
    abifiles =  [f["path"] for f in grouped_files]
    print("*************************************************the abi files are**************************", abifiles)
 
    master_scene = Scene(filenames = {'abi_l1b':abifiles})
    master_scene.load(ABI_channels)
  
    print(master_scene.available_dataset_names())
    print("datasets are now loaded")
    
    log.info(f'Cropping nan edges of {blue}{abi_dt}{reset}')
    t = time.time()
    data = crop.crop_nan_edges_GOES(master_scene, all_channels, all_channels)
    #data = (master_scene, all_channels, all_channels)
    log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    data['channels'] = list(data)
    data['filenames'] = abifiles 
    data["datetime"] = abi_dt
    return data   
    
    
def main(args):
    """Scan for Cbands and get DTG to match up and processes colocation on."""
    abi_dir = Path(args.abi_dir).resolve()
    GOESpath = Path(args.abi_dir).resolve() 
    #master DTG is taken from what GOES files we have
    #match up the GOES ABI to the DNB
    ABI_dict = group_abi_by_time_sat(abi_dir)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>abi_dict is>>>>>>>>>>>>>>", ABI_dict)
    
    # make single file DTG list/dic of the DTGname and the Cband so satpy/colocate etc gets called using this list of DTGs
    matched = ABI_dict
    MATCHED = sorted(matched)
   # print("*******matched is", matched)
    #print("^^^^^^^^^^^matched keys are", matched.keys())
    print("*************************this is MATCHED***************/n", MATCHED)
    keylist = []
    for key in MATCHED:
        keylist.append(key)
    #print("the keylist is", keylist) 
    
    #would love to make this a better looper so i can adjust the # of timestamps permitted based on space and FD vs MESO vs CONUS etc   20FD  timestamps is 25G going into FNNpredict
    datas = []
    dcount = 0
    for datetime in sorted(matched):#[0:100]:
        log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
        #datas.append(process_set_ABI(matched[datetime], dcount, len(matched)))
        try:
            datas.append(process_set_ABI(matched[datetime], dcount, len(matched)))
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
    case1['ABItimes'] = [d['datetime'] for d in datas]
    
    filename = GOESpath / f'{GOESpath.name}_FOG29JAN23_MESO_ALL.npz'
    log.info(f'Writing {blue}{filename.name}{reset}\n' +
             f'{orange}Channels{reset} {channels}\n{orange}')
    np.savez_compressed(filename, **case1)
    log.info(f'Wrote {blue}{filename.name}{reset}')       
    del(case1, datas)
    gc.collect()
            
#      ####CASE 2#####      
#     datas2 = []
#     dcount2 = 0
#     for datetime in sorted(matched)[100:200]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         try:
#             datas2.append(process_set_ABI(matched[datetime], dcount2, len(matched)))
#             dcount2 += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount2} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount2} due to an incomplete dataset")
#             continue
            
#     channels = datas2[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case2 = {c: np.stack(tuple(d[c] for d in datas2)) for c in channels}
#     case2['channels'] = channels
#     case2['samples'] = [d["datetime"] for d in datas2]
#     case2['ABItimes'] = [d['datetime'] for d in datas2]
    
#     filename = GOESpath / f'{GOESpath.name}__FOG29JAN23_MESO_100-200.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#              f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case2)
#     log.info(f'Wrote {blue}{filename.name}{reset}')      
#     del(case2, datas2)
#     gc.collect()
                  
#      #CASE #3       
#     datas3 = []
#     dcount3 = 0
#     for datetime in sorted(matched)[200:300]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         try:
#             datas3.append(process_set_ABI(matched[datetime], dcount3, len(matched)))
#             dcount3 += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount3} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount3} due to an incomplete dataset")
#             continue
                
#     channels = datas3[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case3 = {c: np.stack(tuple(d[c] for d in datas3)) for c in channels}
#     case3['channels'] = channels
#     case3['samples'] = [d["datetime"] for d in datas3]
#     case3['ABItimes'] = [d['datetime'] for d in datas3]
    
#     filename = GOESpath / f'{GOESpath.name}__FOG29JAN23_MESO_200-300.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#              f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case3)
#     log.info(f'Wrote {blue}{filename.name}{reset}')      
#     del (case3, datas3)
#     gc.collect()
#     ########################            
#        ##CASE 4##    
    
#     datas4 = []
#     dcount4 = 0
#     for datetime in sorted(matched)[300:400]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         try:
#             datas4.append(process_set_ABI(matched[datetime], dcount4, len(matched)))
#             dcount4 += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount4} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount4} due to an incomplete dataset")
#             continue
    
#     channels = datas4[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case4 = {c: np.stack(tuple(d[c] for d in datas4)) for c in channels}
#     case4['channels'] = channels

#     case4['samples'] = [d["datetime"] for d in datas4]
#     case4['ABItimes'] = [d['datetime'] for d in datas4]
    
#     filename = GOESpath / f'{GOESpath.name}__FOG29JAN23_MESO_300-400.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#              f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case4)
#     log.info(f'Wrote {blue}{filename.name}{reset}')
#     del (case4, datas4)
#     gc.collect()
    
#     #case5#
#     datas5 = []
#     dcount5 = 0
#     for datetime in sorted(matched)[400:500]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         try:
#             datas5.append(process_set_ABI(matched[datetime], dcount5, len(matched)))
#             dcount5 += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount5} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount5} due to an incomplete dataset")
#             continue   
    
#     channels = datas5[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case5 = {c: np.stack(tuple(d[c] for d in datas5)) for c in channels}
#     #case = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
#     case5['channels'] = channels
 
#     case5['samples'] = [d["datetime"] for d in datas5]
#     case5['ABItimes'] = [d['datetime'] for d in datas5]
    
#     #filename2 = /gdata5/cpasilla/DATA/f'{path.name}_RefGOES_case.npz'
#     filename = GOESpath / f'{GOESpath.name}__FOG29JAN23_MESO_400-500.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#              f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case5)
#     log.info(f'Wrote {blue}{filename.name}{reset}')
#     del (case5, datas5)
#     gc.collect()

#     ###
#     datas = []
#     dcount = 0
#     for datetime in sorted(matched)[500:-1]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         datas.append(process_set_ABI(matched[datetime], dcount, len(matched)))
#         try:
#             datas.append(process_set_ABI(matched[datetime], dcount, len(matched)))
#             dcount += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount} due to an incomplete dataset")
#             continue
            
#     channels = datas[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case1 = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
#     case1['channels'] = channels
#     case1['samples'] = [d["datetime"] for d in datas]
#     case1['ABItimes'] = [d['datetime'] for d in datas]
    
#     filename = GOESpath / f'{GOESpath.name}__FOG29JAN23_MESO_500on.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#             f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case1)
#     log.info(f'Wrote {blue}{filename.name}{reset}')       
#     del(case1, datas)
#     gc.collect()
            
     ###CASE 2#####      
#     datas2 = []
#     dcount2 = 0
#     for datetime in sorted(matched)[60:-1]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         try:
#             datas2.append(process_set_ABI(matched[datetime], dcount2, len(matched)))
#             dcount2 += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount2} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount2} due to an incomplete dataset")
#             continue
            
#     channels = datas2[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case2 = {c: np.stack(tuple(d[c] for d in datas2)) for c in channels}
#     case2['channels'] = channels
#     case2['samples'] = [d["datetime"] for d in datas2]
#     case2['ABItimes'] = [d['datetime'] for d in datas2]
    
#     filename = GOESpath / f'{GOESpath.name}_DEC15_CC_GOES_case_60on.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#             f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case2)
#     log.info(f'Wrote {blue}{filename.name}{reset}')      
#     del(case2, datas2)
#     gc.collect()
                  
# #      #CASE #3       
# #     #datas3 = []
# #     #dcount3 = 0
# #     #for datetime in sorted(matched)[140:160]:
# #      #   log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#       #  try:
#        #     datas3.append(process_set_ABI(matched[datetime], dcount3, len(matched)))
#         #    dcount3 += 1
                         
#         #except KeyError:
#          #   log.info(f"skipped {dcount3} as there was no matching data")            
#           #  continue
    
#         #except IndexError:
#          #   log.info(f"skipped {dcount3} due to an incomplete dataset")
#           #  continue
                
#     #channels = datas3[0]['channels']
#     #print("the channels are", channels)
#     #print("starting the stacking")
#     #case3 = {c: np.stack(tuple(d[c] for d in datas3)) for c in channels}
#     #case = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
#     #case3['channels'] = channels
#     #case3['samples'] = [d["datetime"] for d in datas3]
#     #case3['ABItimes'] = [d['datetime'] for d in datas3]
    
#     #filename = GOESpath / f'{GOESpath.name}_SSSR_CONUS_GOES_case_140-160.npz'
#     #log.info(f'Writing {blue}{filename.name}{reset}\n' +
#      #        f'{orange}Channels{reset} {channels}\n{orange}')
#     #np.savez_compressed(filename, **case3)
#     #log.info(f'Wrote {blue}{filename.name}{reset}')      
#     #del (case3, datas3)
#     #gc.collect()
#     ########################            
#        ##CASE 4##    
    
#     #datas4 = []
#     #dcount4 = 0
#     #for datetime in sorted(matched)[160:180]:
#      #   log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#       #  try:
#        #     datas4.append(process_set_ABI(matched[datetime], dcount4, len(matched)))
#         #    dcount4 += 1
                         
#         #except KeyError:
#          #   log.info(f"skipped {dcount4} as there was no matching data")            
#           #  continue
    
#         #except IndexError:
#          #   log.info(f"skipped {dcount4} due to an incomplete dataset")
#           #  continue
    
#     #channels = datas4[0]['channels']
#     #print("the channels are", channels)
#     #print("starting the stacking")
#     #case4 = {c: np.stack(tuple(d[c] for d in datas4)) for c in channels}
#     #case = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
#     #case4['channels'] = channels

#     #case4['samples'] = [d["datetime"] for d in datas4]
#     #case4['ABItimes'] = [d['datetime'] for d in datas4]
    
#     #filename = GOESpath / f'{GOESpath.name}_CC_GOES_case_160-180.npz'
#     #log.info(f'Writing {blue}{filename.name}{reset}\n' +
#      #        f'{orange}Channels{reset} {channels}\n{orange}')
#     #np.savez_compressed(filename, **case4)
#     #log.info(f'Wrote {blue}{filename.name}{reset}')
#     #del (case4, datas4)
#     #gc.collect()
    
#     #case5#
#     datas5 = []
#     dcount5 = 0
#     for datetime in sorted(matched)[180:200]:
#         log.info(f"the DTG is, {datetime}, and matched[datetime] is {matched[datetime]}")
#         try:
#             datas5.append(process_set_ABI(matched[datetime], dcount, len(matched)))
#             dcount5 += 1
                         
#         except KeyError:
#             log.info(f"skipped {dcount5} as there was no matching data")            
#             continue
    
#         except IndexError:
#             log.info(f"skipped {dcount5} due to an incomplete dataset")
#             continue   
    
#     channels = datas5[0]['channels']
#     print("the channels are", channels)
#     print("starting the stacking")
#     case5 = {c: np.stack(tuple(d[c] for d in datas5)) for c in channels}
#     #case = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
#     case5['channels'] = channels
 
#     case5['samples'] = [d["datetime"] for d in datas5]
#     case5['ABItimes'] = [d['datetime'] for d in datas5]
    
#     #filename2 = /gdata5/cpasilla/DATA/f'{path.name}_RefGOES_case.npz'
#     filename = GOESpath / f'{GOESpath.name}_CC_GOES_case_180-200.npz'
#     log.info(f'Writing {blue}{filename.name}{reset}\n' +
#              f'{orange}Channels{reset} {channels}\n{orange}')
#     np.savez_compressed(filename, **case5)
#     log.info(f'Wrote {blue}{filename.name}{reset}')
#     del (case5, datas5)
#     gc.collect()


