#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#if __name__ == "__main__":
 #   main()
    #find_timesteps()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from collections import defaultdict
from glob import glob
from netCDF4 import Dataset
from satpy import Scene
from satpy import available_readers
from satpy import available_writers
from satpy import resample
import math
import datetime as date
from pathlib import Path


spath = '/gdata5/cpasilla/GOESABICOLOCATION_TEST/GOES'

rawpath = spath
spath2= spath +'COLOCATED/RAW_'

#this is designed to go through file directory and find the unique times of data 
# aka "samples" that will be used to group the channel data by              
    
#parts[0] GEO
#parts[1] Channel
#parts[2] satellite
#parts[3] START TIME
#parts[4] END TIME    
#OR_ABI-L1b-RadF-M6C07_G17_s20201560830320_e20201560839398_c20201560839443.nc


def missing_timesteps(sampleset):
    bad_files =[]
    good_sampleset ={}
    for starttime,my_files in sampleset.items():
        if len(my_files)!=4:
            bad_files.append(starttime)
            print('missing data files for', starttime)
            continue
        else:
            good_sampleset[starttime] = my_files
    print("found", len(bad_files),"bad data sets. go to  badfiles.txt to review")
    with open(spath + "badfiles.txt", "w") as f:
        f.write(str(bad_files))
        f.write('\n')
    return good_sampleset

def find_timesteps(path):
    paths = glob(path/f'{path.name}O*')
      #filename = path / f'{path.name}_J01_RefALL_case.npz'
    print(paths)
    sampleset = defaultdict(list)
    for path in paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        starttime = parts[3]
        sampleset[starttime].append(path)
    print ("these are the # of samples by DTG", len(sampleset))
    return missing_timesteps(sampleset)   
      
    
# cropping functions


def croptomultiples(casearray):
    rows=casearray.shape[1]
    columns=casearray.shape[2]
    newrows= rows- (rows % multiples)
    newcolumns=columns - ( columns % multiples)
    finalarray=casearray[:, 0:newrows, 0:newcolumns, :]
    return finalarray

def find9999front(arow):
    for i in range(len(arow)):
        x=arow[i]
        if not math.isclose(x,-9999):
            return i
    return 0 
    
def find9999back(arow):
    for i in range(1,len(arow)):
        x=arow[-i]
        if not math.isclose(x,-9999):
            return i
    return 0

def findXD9999(arrayXD):
    front=[]
    back=[]
    for row in arrayXD:
        front.append(find9999front(row))
        back.append(find9999back(row))
        #print(len(front))
    print('the front position is',max(front), 'the back position is', max(back))  
    return max(front), max(back) 
    
def cropdeadspace(single):    
    frontposition=[]
    backposition=[]
    channels= [single[:,:,i] for i in range(0,single.shape[-1])]
    for c in channels:
        maxfront,maxback = findXD9999(c)
        frontposition.append(maxfront)
        backposition.append(maxback)
    maxfrontposition = max(frontposition)
    maxbackposition = max(backposition)
    print('the front position is',maxfrontposition, 'the back position is', maxbackposition) 
    rowlength=single.shape[-2]
    croppedsingle=single[:,maxfrontposition:rowlength-maxbackposition,:]
    return croppedsingle, maxfrontposition, maxbackposition

def finddims(arrays):
    ys=[a.shape[0] for a in arrays]   
    xs=[a.shape[1] for a in arrays]   
    print("Y is", min(ys), max(ys), "X is", min(xs),max (xs))
    return min(ys), max(ys), min(xs),max (xs)

def croprawarray(sampleoutput):
    ymin, ymax, xmin, xmax = finddims(sampleoutput.values())
    FRS={starttime: a[0:ymin,0:xmin] for starttime, a in sampleoutput.items()}
    return FRS

# this is designed to pull out the group of files that occur at the same time 
#"sampleset" to run through the colocation and channel combining process for each timestep
def processonesample(starttime,my_files):
   
    
#def processonesamplearray (starttime,my_files)
    #upload ABI data files 
    rawC07=Dataset(spath2 + "G*C07*" + starttime + "*.nc")
    rawC11=Dataset(spath2 + "G*C11*" + starttime + "*.nc")
    rawC13=Dataset(spath2 + "G*C13*" + starttime + "*.nc")
    rawC15=Dataset(spath2 + "G*C15*" + starttime + "*.nc")
   

    # get info on raw data the colocated and original formats
    #Nan--> missing #array.filled(-9999) replaces missing values
    #pull out lat/long
    #fill all the arrays 
    filleddic = {}

    filleddic['FLat'] =  rawC07['latitude'][:].filled(-9999)
    filleddic['FLong'] = rawC07['longitude'][:].filled(-9999)

    filleddic['FrawC07'] = rawC07['C07'][:].filled(-9999)
    filleddic['FrawC11'] = rawC11['C11'][:].filled(-9999)
    filleddic['FrawC13'] = rawC13['C13'][:].filled(-9999)
    filleddic['FrawC15'] = rawC15['C15'][:].filled(-9999)

    #print initial max/min values before the -9999

    for name, array in filleddic.items(): 
        print (f'{name} max is {np.max(array)} and min is {np.min(array)}')

    for name, array in filleddic.items():       
        Rma=array[array != -9999]
        Rma.max()
        Rma.min()
        A=(array>0).sum()
        N=(array==-9999).sum()
        D=N+A

        print(name+ "max is", np.max(Rma), "min is", np.min(Rma))
        print(name + "has this many valid", A,)
        print(name + "has this many null values", N,)
        print(name + "has this many  values", D,)

   
    print('combine to a single 3D array')

    for name, array in filleddic.items():   
        print (f'{name} shape is {array.shape}') 
    # we have dic of numpy arrays for each colocated file

    global colocateddic
    colocateddic = {}
    for name, array in filleddic.items(): 
        if "raw" not in name:
            colocateddic[name] = array
    print('colocated dictionary keys are', colocateddic.keys())

    dicorder=['FLat', 'FLong', 'FrawC07', 'FrawC11', 'FrawC13', 'FrawC15'] 
    arrays=[colocateddic[k] for k in dicorder]
    single=np.stack(tuple(arrays),axis=-1) #combines
    print(single.shape)
    nodeadspacesingle, cropleft, cropright= cropdeadspace(single)
    return nodeadspacesingle


#processes all the samples in the case
def processallsamples(sampleset,limit=None):
    if limit is None:
        limit=len(sampleset)
    global sampleoutput
    sampleoutput={}
    for starttime,myfiles in sampleset.items():
        if limit == 0:
            break
        else:
            limit = limit-1
        print( (len(sampleset)-limit),"/",len(sampleset))
        print("******************started", starttime, "*******************************")
          #for i in len(sampleset):
        if len(myfiles)!=4:
            print('missing data files for', starttime)
            continue
        else:
        #open the files in the dic
            x = processonesample(starttime,myfiles)# NODEADSPACESINGLE IS THIS X
            sampleoutput[starttime] = x
        print("***************completed processing of", starttime, "*******************************")
    #crops to the smallest
    CROPPEDdict=croprawarray(sampleoutput)  
    #stack them
    SINGLECASEARRAY=np.stack(tuple(CROPPEDdict.values())) #combines
    print(SINGLECASEARRAY.shape)
    # now reduce to the 256x256y size
    return croptomultiples(SINGLECASEARRAY), CROPPEDdict   

def main(args): 
    path = Path(args.abi_dir).resolve()
    print('path is', path)
    SINGLE_CASE_ARRAY, CROPPEDdict = processallsamples(find_timesteps(path))#, limit=3)
#save
    import time
    timestr = time.strftime("%Y%m%d-%H%M")
    np.save(path + "RAW_MASTER_ARRAY_GOESONLY" + timestr, SINGLE_CASE_ARRAY)
#array colocated to DNB, with no -9999 edges, all same size to smallest dims, all channles,all samples



#if __name__ == "__main__":
 #   main(args.abi_dir)
    #find_timesteps()
    

def nan_count(row):
    ct = sum(1 for _ in it.takewhile(np.isnan, row))
    return ct if ct < len(row) / 2 else 0
def pairwise_max(n, m):
    return max(n[0], m[0]), max(n[1], m[1])
def pairwise_min(n, m):
    return min(n[0], m[0]), min(n[1], m[1])
def nan_edges(rows):
    """For each row, count the consecutive NaN blocks at the start and
    the end of the row. Then aggregate the max NaN count across all rows."""
    counts = ((nan_count(r), nan_count(r[::-1])) for r in rows)
    return ft.reduce(pairwise_max, counts)
def arrays_and_edges(scn: Scene):
    """Given a colocated scene, gather longitude and latitude arrays and
    sensor arrays. And return the maximum NaN edges found in each channel."""
    ae = [(arr, nan_edges(arr)) for arr in (scn[c].values for c in all_channels)]
    arrs = [scn[c].values for c in lat_long_both[:2]] + [a[0] for a in ae]
    return arrs, [a[1] for a in ae]
def crop_nan_edges(scn: Scene):
    """Using the maximum NaN edges found across all sensor channels,
    crop all channels to eliminate NaN edges. Return a dict mapping the
    channel names to the numpy arrays."""
    arrs, edges = arrays_and_edges(scn)
    front, back = ft.reduce(pairwise_max, edges)
    till = arrs[0].shape[-1] - back
    return {name: arr[:, front:till] for name, arr in zip(lat_long + all_channels, arrs)}

def process_pair(pair, out_path: Path, filename: Path):
    """Pair is a list of two parsed filenames (see the function parse_filename below).
    Given these two files, use Scene to load the appropriate channels.
    Then save these original channels (.png and optionally .nc files).
    Then resample (colocate) to make the sensor channels match up.
    Then save these colocated channels.
    Crop the NaN edges, tag with meta information (which files were used as input),
    And finally save the numpy arrays (so we don't need to recompute next time)"""
    log.info(f'Colocating {blue}{pair[0]["datetime"]}{reset}')
    scn = Scene(reader='abi_l1b', filenames=[f['path'] for f in pair])
    scn.load(all_channels + lat_long)
    save_datasets(scn, 'ORIGINAL_', str(out_path))

    log.info(f'Cropping nan edges of {blue}{pair[0]["datetime"]}{reset}')
    t = time.time()
    data = crop_nan_edges(save_datasets)
    log.debug(f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds')

    data['channels'] = list(data)
    data['filenames'] = [f['filename'] for f in pair]
    log.info(f'Saving {blue}{filename.name}{reset}')
    np.savez(filename, **data)



def processed_file(pair, out_path: Path, curr_idx, len_pairs):
    """Given a pair of parsed filenames, compute the colocated and NaN cropped array data.
    If this work has previously been done, we can just read the array data from the saved file."""
    f = out_path / (pair[0]['datetime'] + '.npz')
    log.debug(f'{rgb(255,0,0)}Processing{reset} timestep {bold}{curr_idx + 1}/{len_pairs}{reset}')
    if not f.exists():
        process_pair(pair, out_path, f)
    else:
        log.debug(f'Using previously computed {blue}{pair[0]["datetime"]}{reset}')
    return f

def parse_time(time_str):
    return datetime.datetime.strptime(time_str, format_abi)# if len(time_str) /= 14 else format_abi)

def parse_filename_abi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    start = parse_time(n[3][1:-1])
    end = parse_time(n[4][1:-1])
    return {'filename': name, 'channels': n[0], 'satellite': n[2],
            'sat_and_start': f'{n[2]} {start}',
            'start': start, 'end': end, 'path': str(path)}

def group_abi_by_time_sat(nc_dir):
    ncs = [parse_filename_abi(f) for f in nc_dir.iterdir() if f.suffix == '.nc']
    def red(d, nc):
        d[nc['sat_and_start']].append(nc)
        return d
    d = ft.reduce(red, ncs, defaultdict(list))
    return d


def group_by_datetime(ncs):
    """Given a list of parsed filenames of nc files, pair them up by
    corresponding datetime (so that these pairs can be colocated later by the
    Scene library. Filenames that don't have a paired file are returned separately."""
    def f(d, nc):
        d[nc['datetime']].append(nc)
        return d
    d = ft.reduce(f, ncs, defaultdict(list))
    paired, unpaired = {}, {}
    for dt, nc_list in d.items():
        p = paired if len(nc_list) == 4 else unpaired
        p[dt] = nc_list
    return paired, unpaired

def grouped_nc(nc_dir):
    ncs = [parse_filename(f) for f in nc_dir.iterdir() if f.suffix == '.nc']
    return group_by_datetime(ncs)

def pair_with_ncs(nc_dict):
    nc_lists = list(nc_dict.values())
    ret = []
    best = nc_lists
    ret.append(str(best))
    return ret

def pack_case(args):
    """Scan for nc files, pair them by datetime,
    then gather all samples together, crop to the minimum size, and save the entire
    case worth of channel data to a file, case.npz.
    This file, case.npz, contains each of the channels as separate array data.
    It also contains meta information like which channels are included and which h5 files
    went into making this case."""
    nc_dir = Path(args.nc_dir).resolve()
    #h5s = gather_h5s(h5_dir)
    nc_dict = group_abi_by_time_sat(nc_dir)

    paired = pair_with_ncs(nc_dict)
    print(paired)
    paired_sorted = sorted(paired)#, key=lambda h5_ncs: h5_ncs[0]['start'])
    for nc_list in paired_sorted:
        nc_list = sorted(nc_list, key=lambda nc: nc['start'])
        nc_first = nc_list[0]
        #print(f'nc_list first {nc_first}')
        #import sys
        #sys.exit()
        print(f'{bold}{nc_first["start"]} -> {nc_first["end"]}{reset}')
        for nc in nc_list:
            print(f'{nc["filename"]}')
        print()

    print(paired_sorted)
    files = [processed_file(paired[datetime], col, idx, len(paired)) for idx, datetime in enumerate(paired_sorted)]
    npzs = [np.load(f) for f in files]
    min_rows, min_cols = ft.reduce(pairwise_min, [x['C07'].shape for x in npzs])
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
    log.info(f'Wrote {blue}{filename.name}{reset}')





    
    

