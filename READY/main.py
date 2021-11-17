#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import subprocess
import logging
import sqlite3
import time
import numpy as np
import functools as ft
#import tensorflow as tf
import common
from common import log, bold, reset, color, rgb, blue

import Rdatetime
import info
import aoi
#import crop
import combine_case
import patch
import NADIR_crop

import VIIRS_pack_case
#import ABI_pack_case
import VIIRS_trio
import ABI_processing
import btd
import band_norm
import DNB_norm
#import VIIRS_prep
import NAN
import NANrows
#import NANnearest
import NANfill

import I2M_patchnaoi
import I2M_master
#import MLR_train
import FNN_train
#import FNN_retrain
import FNN_predict
#import FNN_assess
#import FNN_PnA

#import normalize
#import learning_prep
#import model_validation
#import MLR_postprocess
#import INPUT2MODEL


def shell_setup():
    bin_dir = Path.home() / 'bin'
    cmd_symlink = bin_dir / 'night2day'
    main_file = Path(__file__).resolve()
    profile = Path.home() / '.profile'
    zshrc = Path.home() / '.zshrc'
    x = 'export PATH="$HOME/bin:$PATH"'
# TODO shell setup
#   create $HOME/bin folder if not already exists
#   symlink $HOME/bin/night2day -> __file__ (if not already there)
#   add to PATH (if not already there) by adding to .profile and .zshrc
#   export PATH="$HOME/bin:$PATH"


def status(args):
    log.info("here")
def learning_cmd(args):
    db_path = common.locate_db(None)
    learning_prep.learning_prep(db_path)
def model_val_cmd(args):
    db_path = common.locate_db(None)
    model_validation.model_val(db_path)
def MLR_cmd(args):
    MLR_SKL.MLR(args.npzfilename, args.nick)
def scatter_cmd(args):
    scatter.scatter(args.npzfilename, args.nick, args.samplesize)      

#def combine_cmd(args):
#    log.info(f'Starting combine cases')
#    output = combine_case.combine_cases([np.load(p) for p in args.npz_path])
#    log.info(f'Writing {blue}{args.npz_path[0]}.npz{reset}')
#    np.savez(args.outputname, **output)
#   log.info(f'Wrote {blue}{args.npz_path[0]}.npz{reset}')

# night2day status [case root dir | night2day.db file]
# night2day info file
# night2day log [root | db file]
# night2day pack-case [root | db | cwd]
# night2day normalize
# night2day prep-learning

desc = f'''Process satellite images for machine learning.'''
parser = argparse.ArgumentParser(description=desc)
parser.set_defaults(func=status)
subparsers = parser.add_subparsers()

msg = (f'Pack a case into a single array',
       '''Process and pack a case into a single array.
       Matches time-correlated images, regularizes dimensions across all images.''')


######high level
info_p = subparsers.add_parser('info', help=' gives npz file summary info') 
info_p.set_defaults(func=info.main)
info_p.add_argument('npz_path', help='Path to npz file')
info_p.add_argument('-q', '--quiet', action='count', default=0)

### DTG stuff
##### PACKCASES######
Rdatetime_p = subparsers.add_parser('RDT', help=' makes .txt file of the DTGs for use in the Reflectance work')
Rdatetime_p.set_defaults(func=Rdatetime.main)
Rdatetime_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
Rdatetime_p.add_argument('-q', '--quiet', action='count', default=0)


##### PACKCASES######

VIIRS_trio_p = subparsers.add_parser('VIIRS-trio', help=' makes the numpy array with the DNB, Mband and SM_Reflectance')
VIIRS_trio_p.set_defaults(func=VIIRS_trio.pack_case)
VIIRS_trio_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
VIIRS_trio_p.add_argument('-q', '--quiet', action='count', default=0)

ABI_processing_p = subparsers.add_parser('ABI-processing', help=' makes the numpy array with the DNB, Cband, Mband and SM_Reflectance')
ABI_processing_p.set_defaults(func=ABI_processing.main)
ABI_processing_p.add_argument('viirs_dir')
ABI_processing_p.add_argument('abi_dir')
ABI_processing_p.add_argument('-q', '--quiet', action='count', default=0)


VIIRS_pack_case_p = subparsers.add_parser('VIIRS-packcase', help=msg[0], description=msg[1])
VIIRS_pack_case_p.set_defaults(func=VIIRS_pack_case.pack_case)
VIIRS_pack_case_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
VIIRS_pack_case_p.add_argument('--save-images', action='store_true', help='Should save image files')
VIIRS_pack_case_p.add_argument('-q', '--quiet', action='count', default=0)

#ABI_pack_case_p = subparsers.add_parser('ABI-pack-case')
#ABI_pack_case_p.set_defaults(func=ABI_pack_case.pack_case)
#ABI_pack_case_p.add_argument('h5_dir')
#ABI_pack_case_p.add_argument('nc_dir')
#ABI_pack_case_p.add_argument('--save-images', action='store_true', help='Should save image files')
#ABI_pack_case_p.add_argument('-q', '--quiet', action='count', default=0)
    
    
comb_p = subparsers.add_parser('combine-cases', help='combine multiple cases identified in cmd line')
comb_p.set_defaults(func=combine_case.main)
comb_p.add_argument('-q', '--quiet', action='count', default=0)
comb_p.add_argument('--outputname', default = 'COMBINED.npz', help ='the name of new combined file')
comb_p.add_argument('npz_path', nargs='+', help='npz files to combine')

####MODIFY DATA

btd_p = subparsers.add_parser('btd', help='makes the desired BTD channels and norms, defaults to all unless specified')
btd_p.set_defaults(func=btd.btd)
btd_p.add_argument('npz_path', help='Path to npz file')
#VIIRS_btd_p.add_argument('--BTDs', default= None, help='what BTDs we want')
btd_p.add_argument('-q', '--quiet', action='count', default=0)

band_norm_p = subparsers.add_parser('band-norm', help='Normalizes  any M and C bands if present') 
band_norm_p.set_defaults(func=band_norm.main)
band_norm_p.add_argument('npz_path', help='Path to npz file')
band_norm_p.add_argument('-q', '--quiet', action='count', default=0)

DNB_norm_p = subparsers.add_parser('DNB-norm', help='Normalizes the DNB bands to the various truth options') 
DNB_norm_p.set_defaults(func=DNB_norm.DNBnorm)
DNB_norm_p.add_argument('npz_path', help='Path to npz file')
DNB_norm_p.add_argument('-q', '--quiet', action='count', default=0)

#VIIRS_prep_p = subparsers.add_parser('VIIRS-prep', help='combines the DNBnorm, BTDprocess and band norm to one step')
#VIIRS_prep_p.set_defaults(func=VIIRS_prep.prep)
#VIIRS_prep_p.add_argument('npz_path', help='Path to npz file')
#VIIRS_prep_p.add_argument('-q', '--quiet', action='count', default=0)


NAN_p = subparsers.add_parser('NAN', help='Remove all patches with ANY NANs-- only use after patched')
NAN_p.set_defaults(func=NAN.NAN)
NAN_p.add_argument('npz_path', help='Path to npz file')
NAN_p.add_argument('--keep', action = 'store_true', help='removed NANs on case unless flagged otherwise')
NAN_p.add_argument('-q', '--quiet', action='count', default=0)

NANrows_p = subparsers.add_parser('NANrows', help='Remove all samples with NAN rows--')
NANrows_p.set_defaults(func=NANrows.NAN)
NANrows_p.add_argument('npz_path', help='Path to npz file')
NANrows_p.add_argument('--keep', action = 'store_true', help='removed rows of NANs and filled random NANS w 0 on case unless flagged otherwise')
NANrows_p.add_argument('--both', action = 'store_true', help='if not called does individuallys, if called will run and save both the norow and yesrows')
NANrows_p.add_argument('-q', '--quiet', action='count', default=0)


NANfill_p = subparsers.add_parser('NANfill', help='Replace all NANs w 9999')
NANfill_p.set_defaults(func=NANfill.NANfill)
NANfill_p.add_argument('npz_path', help='Path to npz file')
NANfill_p.add_argument('-q', '--quiet', action='count', default=0)

#NANnearest_p = subparsers.add_parser('NANnearest', help='Replace NANs with nearest vertical value')
#NANnearest_p.set_defaults(func=NANnearest.NANnearest)
#NANnearest_p.add_argument('npz_path', help='Path to npz file')
#NANnearest_p.add_argument('-q', '--quiet', action='count', default=0)

patch_p = subparsers.add_parser('patch', help='Cut into smaller patches')
patch_p.set_defaults(func=patch.patch)
patch_p.add_argument('npz_path', help='Path to npz file')
patch_p.add_argument('--PATCHSIZE', default= 256, type=int, help='desired patchsize')
patch_p.add_argument('-q', '--quiet', action='count', default=0)

aoi_p = subparsers.add_parser('aoi', help='Filter based of Area of Interest')
aoi_p.set_defaults(func=aoi.aoi)
#aoi_p.add_argument('--pixel', action = 'store_true' )
aoi_p.add_argument('npz_path', help='Path to npz file')
aoi_p.add_argument('NSEW', type=int, nargs=4, help='NSEW bounding box')
aoi_p.add_argument('--pixel', action = 'store_true' )
aoi_p.add_argument('-q', '--quiet', action='count', default=0)

NADIR_crop_p = subparsers.add_parser('NADIR', help=' reduces array to NADIR + 600') 
NADIR_crop_p.set_defaults(func=NADIR_crop.main)
NADIR_crop_p.add_argument('npz_path', help='Path to npz file')
NADIR_crop_p.add_argument('-q', '--quiet', action='count', default=0)


######### combine patch, NAN patch removal and AOI into oen for training data prep
I2M_patchnaoi_p = subparsers.add_parser('I2M-PNA', help='Patch and AOI boundry confirmation')
I2M_patchnaoi_p.set_defaults(func=I2M_patchnaoi.patchnaoi)
I2M_patchnaoi_p.add_argument('-q', '--quiet', action='count', default=0)
I2M_patchnaoi_p.add_argument('npz_path', help='Path to npz file')
I2M_patchnaoi_p.add_argument('--aoi', type=int, nargs=4, help='NSEW bounding box')

I2M_master_p = subparsers.add_parser('I2M-master', help='Does I2M functions in the following order NADIR, nanrows ( keeps just those without NANrows), Derive band norms and BTDs removes original data ')
I2M_master_p.set_defaults(func=I2M_master.main)
I2M_master_p.add_argument('-q', '--quiet', action='count', default=0)
I2M_master_p.add_argument('npz_path', help='Path to npz file')


######### MLR steps
#MLR_p = subparsers.add_parser('MLR-train', help='Train a MLR SKL model from .npz data and provide ML output')
#MLR_p.set_defaults(func=MLR_train.MLR)
#MLR_p.add_argument('npz_path', help='Path to npz file')
#MLR_p.add_argument('--DNB', help='the DNB truth we are using')
#MLR_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
#MLR_p.add_argument('-q', '--quiet', action='count', default=0)

######### FNN steps

FNN_train_p = subparsers.add_parser('FNN-train', help='Train a FNN given a set of data')
FNN_train_p.set_defaults(func=FNN_train.FNN_train)
FNN_train_p.add_argument('npz_path', help='Path to npz file')
FNN_train_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
FNN_train_p.add_argument('-q', '--quiet', action='count', default=0)

#FNN_retrain_p = subparsers.add_parser('FNN-retrain', help='REtrain a FNN given a predictos/predictand array meant for use after inital train on data')
#FNN_retrain_p.set_defaults(func=FNN_retrain.FNN_retrain)
#FNN_retrain_p.add_argument('npz_path', help='Path to npz file of TORS/TAND')
#FNN_retrain_p.add_argument('-q', '--quiet', action='count', default=0)

FNN_predict_p = subparsers.add_parser('FNN-predict', help='Predict ML values with FNN') #may become predict only) 
FNN_predict_p.set_defaults(func=FNN_predict.predict)
FNN_predict_p.add_argument('npz_path', help='Path to npz file with channels')
FNN_predict_p.add_argument('model_path', help='Path to Model folder')
FNN_predict_p.add_argument('channel_path', help='Path to Model Channels')
FNN_predict_p.add_argument('-q', '--quiet', action='count', default=0)        
                                      
#FNN_assess_p = subparsers.add_parser('FNN-assess', help='compare ML vs DNB values') # may become train only) 
#FNN_assess_p.set_defaults(func=FNN_assess.assessment)
#FNN_assess_p.add_argument('npz_path', help='Path to npz file with DNB and band raw values')
#FNN_assess_p.add_argument('norm_path', help='Path to npz file with normed values')
#FNN_assess_p.add_argument('ML_path', help='Path to ML data')
#FNN_assess_p.add_argument('-q', '--quiet', action='count', default=0)   


####### old files ####### 
#norm_p = subparsers.add_parser('normalize', help='Normalize and derive channels')
#norm_p.set_defaults(func=normalize.normalize)
#norm_p.add_argument('npz_path', help='Path to npz file')
#norm_p.add_argument('-q', '--quiet', action='count', default=0)

#learn_p = subparsers.add_parser('learn-prep', help='Create file for input to learning')
#learn_p.set_defaults(func=learning_cmd)
#learn_p.add_argument('-q', '--quiet', action='count', default=0)

#model_val_p = subparsers.add_parser('model-val', help='Create  validation file for model validation')
#model_val_p.set_defaults(func=model_val_cmd)
#model_val_p.add_argument('-q', '--quiet', action='count', default=0)

#MLR_post = subparsers.add_parser('MLR-post', help='take npz and model and make final data sets')
#MLR_post.set_defaults(func=MLR_postprocess.postprocess)
#MLR_post.add_argument('-q', '--quiet', action='count', default=0)
#MLR_post.add_argument('npz_path', help='Path to npz file')
#MLR_post.add_argument('model_path', help='Path to model .pickle file')
#MLR_post.add_argument('nick', help='Name to create new folder structure')

#INPUT2MODEL_p = subparsers.add_parser('I2M', help='turns channels arrays to TORS/TAND') 
#INPUT2MODEL_p.set_defaults(func=INPUT2MODEL.FINALprep)
#INPUT2MODEL_p.add_argument('npz_path', help='Path to npz file')
#INPUT2MODEL_p.add_argument('-q', '--quiet', action='count', default=0)

"""The default is to display both log.info() and log.debug() statements.
But if the user runs this program with the -q or --quiet flags, then only
display the log.info() statements."""
args = parser.parse_args()
if hasattr(args, 'quiet') and args.quiet > 0:
    log.setLevel(logging.INFO)

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))    
starttime= time.ctime()
print(f'Starting {bold}{starttime}{reset}')
tstart = time.time()
args.func(args)
print(f'Started at  {bold}{starttime}{reset}')
print(f'Finished at {bold}{time.ctime()}{reset}')
tend = time.time()
#print(f'Took {bold}{tend-tstart:.2f}{reset} seconds to run')
secondspast = tend-tstart
print(f'Took {bold}{secondspast/60:.2f}{reset} minutes to run')



