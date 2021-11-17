#Open VIIRS get time stamps save as .txt file

import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict
import sys
import common
from common import log, rgb, reset, blue, orange, bold

# File name: GDNBO-SVDNB_j01_d20200110_t1031192_e1036592_b*
def parse_filename(path):
    """Given the path to one of the raw satellite sensor .h5 files, parse the
    file name into separate fields, for ease of use in later logic."""
    n = path.name.split('_')
    #return {'filename': path.name, 'channels': n[0], 'satellite': n[1],
     #       'date': n[2], 'start': n[3], 'end': n[4], 'b':n[5], 'datetime': n[2] + '_' + n[3],
      #      'Rdatetime': n[2] + '_' + n[3]+ '_'+n[4]+'_'+n[5], 'path': str(path)}
    #return {'Rdatetime': n[2] + '_' + n[3]+ '_'+n[4]+'_'+n[5]}
    Rdatetime = n[2] + '_' + n[3]+ '_'+n[4]+'_'+n[5]
    return Rdatetime

#def group_by_datetime(h5s):
 #   """Given a list of parsed filenames of h5 files, pair them up by
  #  corresponding datetime (so that these pairs can be colocated later by the
   # Scene library. Filenames that don't have a paired file are returned separately."""
    #def f(d, h5):
     #   d[h5['Rdatetime']].append(h5)
      #  return d
    #d = ft.reduce(f, h5s, defaultdict(list))
    #paired, unpaired = {}, {}
    #for dt, h5_list in d.items():
     #   p = paired if len(h5_list) == 2 else unpaired
      #  p[dt] = h5_list
    #return paired, unpaired


def grouped_h5s(h5_dir):
    h5s = [parse_filename(f) for f in h5_dir.iterdir() if f.suffix == '.h5']
    #return group_by_datetime(h5s)
    #print (h5s)
    return h5s

#def Rdatetime (args):
    #path = Path(args.h5_dir).resolve()
    l#og.info(f'h5_dir path is {path}')
 #   h5s = [parse_filename(f) for f in h5_dir.iterdir() if f.suffix == '.h5']
  #  np.savetxt('R_datetime.txt',h5s)
   # return h5s
def main(args):
    path = Path(args.h5_dir).resolve()
    log.info(f'h5_dir path is {path}')
    #DTGpaired, DTGunpaired = grouped_h5s(path)
    DTGs = grouped_h5s(path)
    print(len(DTGs))
    
    DTGsolo = []
    for i in DTGs:
        if i not in DTGsolo:
            DTGsolo.append(i)
    numfiles = [(len(DTGsolo))]
    endfile = numfiles + DTGsolo

    print(endfile)
    print(len(DTGsolo))  
    print(len(endfile))
    #print(DTGs)
    #np.savetxt('R_datetimesolo.txt', [len(DTGsolo), DTGsolo], fmt='%s', newline='\n')
    np.savetxt('Reflectance_datetimes.txt', endfile, fmt='%s', newline='\n')
    
   
    