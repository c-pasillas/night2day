
import numpy as np
import itertools as it
import functools as ft
from pathlib import Path
import sqlite3
import time
from collections import defaultdict
from satpy import Scene
import datetime
import common
from common import log, bold, reset, color, rgb, blue
import sys


viirs_channels = ['DNB', 'M13', 'M14', 'M15', 'M16']
GOES_channels = ['C07', 'C11', 'C13', 'C15'] #M12 not needed
lat_long_both = ['dnb_latitude', 'dnb_longitude', 'm_latitude', 'm_longitude']
lat_long = ['latitude', 'longitude']
SM_reflectance = ['SM_reflectance']
all_channels = viirs_channels + SM_reflectance


VIIRS_scene = satpy.Scene(filenames=glob('/VIIRS/G*t0848*.h5'),

                       reader='viirs_sdr')

VIIRS_scene.load(['viirs_channels + lat_long_both'])


GOES_scene = satpy.Scene(filenames=glob('/GOES/O1560830*.nc'),

                        reader='abi_l1b')

GOES_scene.load(['GOES_channels + lat_long'])



from satpy import MultiScene, DataQuery

mscn = MultiScene([VIIRS_scene, GOES_scene])

groups = {DataQuery('alldata_group': ['viirs_channels + lat_long_both + GOES_channels +lat_long ]}

mscn.group(groups)
