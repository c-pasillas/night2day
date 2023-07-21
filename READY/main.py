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

#this file runs and gets quick info on your numpy array file
import info

#this is needed to get the date time to then go run the IDL SM_Reflectance process
import Rdatetime

#####these . py take the hdf and nc files and make the colocated cases
# these .py SHOULD be the two main processing inputs of NOAA data after the SM_Reflectance has been calculated, see next if no reflectance needed. 
import VIIRS_raw_process
import ABI_raw_process # for 0:50
import ABI_raw_process_back
import ABI_raw_process_small #for 50:+
import ABI_only
import ABI_only_FDR
import AHI_FullDisk
import AHI_CROP
#import GOES_pack_case
#import workingABIcolocate
#import GOES2NPZ
#import dealer

# pack_case can be run if no DNB Reflecance data needed, means cases will be imagery use only no stats cuz no truth is possible
#import ABI_pack_case
#import VIIRS_pack_case 

########these are acted upon colocated cases to reduce sizes, remove bad data, focus on AIO, focus on bands of concern etc
import combine_case# combine for more cases or by season, yr, etc
import combine_MLs
import Train_master # runs in this order NADIR, nan rows, band norm, btds
import predict_master # runs the btd and bandnorm but keeps and NAN stuff and doesnt make small to NADIR, best to rep true data set application
import GOES_trng_prep # takes the full set and makes Cbands only to trng input
import NADIR_crop # crops the processed array to NADIR + 600
import aoi # can filter by if the whole patch is in the AOI y/n or only take the pixels and make new array
#import crop
import patch #patches input to desired sizes
import NANrows  #removes and sample that has any rows of NANs  ( used on whole or patches)
import NANfill # fills all NANs with 9999
import btd #makes the desired BTDs from the present bands
import band_norm # normalizes the M and C bands present


#these are the actual training and prediction programs
#import MLR_train
import FNN_prep
import FNN_train
import FNN_GOES_expanded_train
import FNN_retrain
import FNN_predict
import FNN_GOES_predict
import FNN_AHI_predict
import FNN_assess
#import FNN_PnA
#import removecloud
#import map_rawbands
#import map_MLNVI

# these are old programs that may/may noy be helpful

import I2M_patchnaoi
#import normalize
#import learning_prep
#import model_validation
#import MLR_postprocess
#import INPUT2MODEL
#import DNB_norm
#import VIIRS_prep
#import VIIRS_trio
#import NAN
#import NANnearest

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

Rdatetime_p = subparsers.add_parser('RDT', help=' makes .txt file of the DTGs for use in the Reflectance work')
Rdatetime_p.set_defaults(func=Rdatetime.main)
Rdatetime_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
Rdatetime_p.add_argument('-q', '--quiet', action='count', default=0)


##### PACKCASES######

VIIRS_raw_process_p = subparsers.add_parser('VIIRS-raw-process', help=' makes the numpy array with the DNB, Mband and SM_Reflectance')
VIIRS_raw_process_p.set_defaults(func=VIIRS_raw_process.pack_case)
VIIRS_raw_process_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
VIIRS_raw_process_p.add_argument('-q', '--quiet', action='count', default=0)

ABI_raw_process_p = subparsers.add_parser('ABI-raw-process', help=' makes the numpy array with the DNB, Cband, Mband and SM_Reflectance')
ABI_raw_process_p.set_defaults(func=ABI_raw_process.main)
ABI_raw_process_p.add_argument('viirs_dir')
ABI_raw_process_p.add_argument('abi_dir')
ABI_raw_process_p.add_argument('-q', '--quiet', action='count', default=0)

ABI_raw_process_p = subparsers.add_parser('ABI-raw-process-back', help=' makes the numpy array with the DNB, Cband, Mband and SM_Reflectance for DTG 50+')
ABI_raw_process_p.set_defaults(func=ABI_raw_process_back.main)
ABI_raw_process_p.add_argument('viirs_dir')
ABI_raw_process_p.add_argument('abi_dir')
ABI_raw_process_p.add_argument('-q', '--quiet', action='count', default=0)

ABI_raw_process_p = subparsers.add_parser('ABI-raw-process-small', help=' makes the numpy array with the DNB, Cband, Mband and SM_Reflectance for  set of 20 DTGs ')
ABI_raw_process_p.set_defaults(func=ABI_raw_process_small.main)
ABI_raw_process_p.add_argument('viirs_dir')
ABI_raw_process_p.add_argument('abi_dir')
ABI_raw_process_p.add_argument('-q', '--quiet', action='count', default=0)

ABI_only_p = subparsers.add_parser('ABI-only-raw', help=' makes the numpy array with the Cband for  sets of DTGs-must edit size and filenames before processing ')
ABI_only_p.set_defaults(func=ABI_only.main)
ABI_only_p.add_argument('abi_dir')
ABI_only_p.add_argument('-q', '--quiet', action='count', default=0)

AHI_FullDisk_p = subparsers.add_parser('AHI-FD', help=' makes the numpy array with the Bband for sets of DTGs-must edit size and filenames before processing ')
AHI_FullDisk_p.set_defaults(func=AHI_FullDisk.main)
AHI_FullDisk_p.add_argument('ahi_dir')
AHI_FullDisk_p.add_argument('-q', '--quiet', action='count', default=0)

AHI_CROP_p = subparsers.add_parser('AHI-CROP', help=' makes the numpy array with the Bband for sets of DTGs and desired crop box-must edit size and filenames before processing ')
AHI_CROP_p.set_defaults(func=AHI_CROP.main)
AHI_CROP_p.add_argument('ahi_dir')
AHI_CROP_p.add_argument('LATLON', type=int, nargs=2, help='latitdue and longitude image is centered on')
AHI_CROP_p.add_argument('-q', '--quiet', action='count', default=0)

ABI_only_FDR_p = subparsers.add_parser('ABI-only-FDR', help=' makes the numpy array with the Cband for  sets of DTGs for a redcued sie given before-must edit size and filenames before processing ')
ABI_only_FDR_p.set_defaults(func=ABI_only_FDR.main)
ABI_only_FDR_p.add_argument('abi_dir')
ABI_only_FDR_p.add_argument('-q', '--quiet', action='count', default=0)

#dealer_p = subparsers.add_parser('dealer', help=' makes the numpy array with the Cbands')
#dealer_p.set_defaults(func=dealer.pack_case)
#dealer_p.add_argument('abi_dir', help='Path to directory with the GOES abi files')
#dealer_p.add_argument('-q', '--quiet', action='count', default=0)


#VIIRS_pack_case_p = subparsers.add_parser('VIIRS-packcase', help=msg[0], description=msg[1])
#VIIRS_pack_case_p.set_defaults(func=VIIRS_pack_case.pack_case)
#VIIRS_pack_case_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
#VIIRS_pack_case_p.add_argument('--save-images', action='store_true', help='Should save image files')
#VIIRS_pack_case_p.add_argument('-q', '--quiet', action='count', default=0)

#ABI_pack_case_p = subparsers.add_parser('ABI-pack-case')
#ABI_pack_case_p.set_defaults(func=ABI_pack_case.pack_case)
#ABI_pack_case_p.add_argument('h5_dir')
#ABI_pack_case_p.add_argument('nc_dir')
#ABI_pack_case_p.add_argument('--save-images', action='store_true', help='Should save image files')
#ABI_pack_case_p.add_argument('-q', '--quiet', action='count', default=0)

#GOES_pack_case_p = subparsers.add_parser('GOES-pack-case', help = "make the GOES case colocated w VIIRS DNB")
#GOES_pack_case_p.set_defaults(func=GOES_pack_case.main)
#GOES_pack_case_p.add_argument('h5_dir')
#GOES_pack_case_p.add_argument('nc_dir')
#GOES_pack_case_p.add_argument('--save-images', action='store_true', help='Should save image files')
#GOES_pack_case_p.add_argument('-q', '--quiet', action='count', default=0)  
  
#GOES2NPZ_p = subparsers.add_parser('GOES2NPZ', help = "make the GOES2NPZ only case")
#GOES2NPZ_p.set_defaults(func=GOES2NPZ.main)
#GOES2NPZ_p.add_argument('abi_dir')
#GOES2NPZ_p.add_argument('-q', '--quiet', action='count', default=0)    
    
#workingABIcolocate_p = subparsers.add_parser('ABI-coloc', help = "working GOES case")
#workingABIcolocate_p.set_defaults(func=workingABIcolocate.pack_case)
#workingABIcolocate_p.add_argument('abi_dir')
#workingABIcolocate_p.add_argument('--save-images', action='store_true', help='Should save image files')
#workingABIcolocate_p.add_argument('-q', '--quiet', action='count', default=0)   
    
comb_p = subparsers.add_parser('combine-cases', help='combine multiple cases identified in cmd line')
comb_p.set_defaults(func=combine_case.main)
comb_p.add_argument('-q', '--quiet', action='count', default=0)
comb_p.add_argument('--outputname', default = 'COMBINED.npz', help ='the name of new combined file')
comb_p.add_argument('npz_path', nargs='+', help='npz files to combine')


comb_p = subparsers.add_parser('combine-MLs', help='combine multiple ML cases identified in cmd line')
comb_p.set_defaults(func=combine_MLs.main)
comb_p.add_argument('-q', '--quiet', action='count', default=0)
comb_p.add_argument('--outputname', default = 'MLcombined.npz', help ='the name of new combined file')
comb_p.add_argument('npz_path', nargs='+', help='npz files to combine')
####MODIFY DATA

Train_master_p = subparsers.add_parser('TRNG-master', help='TRNG PREP (most ideal data set) --modify fxns in order NADIR, nanrows (keeps DTG without NANrows), Derive band norms and BTDs removes original data ')
Train_master_p.set_defaults(func=Train_master.main)
Train_master_p.add_argument('-q', '--quiet', action='count', default=0)
Train_master_p.add_argument('npz_path', help='Path to npz file')

predict_master_p = subparsers.add_parser('PREDICT-master', help='PREDICT PREP (most operational input data set) --modify fxns in order Derive band norms and BTDs removes original data does not NADIR or remove NANs(rows or individual) works for both Cbands and Mbands ')
predict_master_p.set_defaults(func=predict_master.main)
predict_master_p.add_argument('-q', '--quiet', action='count', default=0)
predict_master_p.add_argument('npz_path', help='Path to npz file')

GOES_trng_prep_p = subparsers.add_parser('GOES-trng', help='GOES TRNG PREP (most operational input data set) --modify fxns in order Derive band norms and BTDs removes original data does not NADIR or remove NANs(rows or individual) removes the Mbands so only Cbands due to memory space')
GOES_trng_prep_p.set_defaults(func=GOES_trng_prep.main)
GOES_trng_prep_p.add_argument('-q', '--quiet', action='count', default=0)
GOES_trng_prep_p.add_argument('npz_path', help='Path to npz file')

NADIR_crop_p = subparsers.add_parser('NADIR', help=' reduces array to NADIR + 600') 
NADIR_crop_p.set_defaults(func=NADIR_crop.main)
NADIR_crop_p.add_argument('npz_path', help='Path to npz file')
NADIR_crop_p.add_argument('-q', '--quiet', action='count', default=0)

patch_p = subparsers.add_parser('patch', help='Cut into smaller patches')
patch_p.set_defaults(func=patch.patch)
patch_p.add_argument('npz_path', help='Path to npz file')
patch_p.add_argument('--PATCHSIZE', default= 512, type=int, help='desired patchsize')
patch_p.add_argument('-q', '--quiet', action='count', default=0)

aoi_p = subparsers.add_parser('aoi', help='Filter based of Area of Interest keep patch in AOI or only keep points in AOI')
aoi_p.set_defaults(func=aoi.aoi)
#aoi_p.add_argument('--pixel', action = 'store_true' )
aoi_p.add_argument('npz_path', help='Path to npz file')
aoi_p.add_argument('NSEW', type=int, nargs=4, help='NSEW bounding box')
aoi_p.add_argument('--pixel', action = 'store_true' )
aoi_p.add_argument('-q', '--quiet', action='count', default=0)

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

btd_p = subparsers.add_parser('btd', help='makes the desired BTD channels and norms, defaults to all unless specified must take from raw band values')
btd_p.set_defaults(func=btd.btd)
btd_p.add_argument('npz_path', help='Path to npz file')
#VIIRS_btd_p.add_argument('--BTDs', default= None, help='what BTDs we want')
btd_p.add_argument('-q', '--quiet', action='count', default=0)

band_norm_p = subparsers.add_parser('band-norm', help='Normalizes  any M and C bands if present') 
band_norm_p.set_defaults(func=band_norm.main)
band_norm_p.add_argument('npz_path', help='Path to npz file')
band_norm_p.add_argument('-q', '--quiet', action='count', default=0)


#NAN_p = subparsers.add_parser('NAN', help='Remove all patches with ANY NANs-- only use after patched')
#NAN_p.set_defaults(func=NAN.NAN)
#NAN_p.add_argument('npz_path', help='Path to npz file')
#NAN_p.add_argument('--keep', action = 'store_true', help='removed NANs on case unless flagged otherwise')
#NAN_p.add_argument('-q', '--quiet', action='count', default=0)


#NANnearest_p = subparsers.add_parser('NANnearest', help='Replace NANs with nearest vertical value')
#NANnearest_p.set_defaults(func=NANnearest.NANnearest)
#NANnearest_p.add_argument('npz_path', help='Path to npz file')
#NANnearest_p.add_argument('-q', '--quiet', action='count', default=0)

#DNB_norm_p = subparsers.add_parser('DNB-norm', help='Normalizes the DNB bands to the various truth options') 
#DNB_norm_p.set_defaults(func=DNB_norm.DNBnorm)
#DNB_norm_p.add_argument('npz_path', help='Path to npz file')
#DNB_norm_p.add_argument('-q', '--quiet', action='count', default=0)

#VIIRS_prep_p = subparsers.add_parser('VIIRS-prep', help='combines the DNBnorm, BTDprocess and band norm to one step')
#VIIRS_prep_p.set_defaults(func=VIIRS_prep.prep)
#VIIRS_prep_p.add_argument('npz_path', help='Path to npz file')
#VIIRS_prep_p.add_argument('-q', '--quiet', action='count', default=0)

#VIIRS_trio_p = subparsers.add_parser('VIIRS-trio', help=' makes the numpy array with the DNB, Mband and SM_Reflectance')
#VIIRS_trio_p.set_defaults(func=VIIRS_trio.pack_case)
#VIIRS_trio_p.add_argument('h5_dir', help='Path to directory with the .h5 files')
#VIIRS_trio_p.add_argument('-q', '--quiet', action='count', default=0)


######### combine  NADIR, patch, and AOI into one for training data prep
I2M_patchnaoi_p = subparsers.add_parser('I2M_patchnaoi', help='reduce data to Patch512 and AOI boundry confirmation')
I2M_patchnaoi_p.set_defaults(func=I2M_patchnaoi.patchnaoi)
I2M_patchnaoi_p.add_argument('-q', '--quiet', action='count', default=0)
I2M_patchnaoi_p.add_argument('npz_path', help='Path to npz file')
I2M_patchnaoi_p.add_argument('--aoi', type=int, nargs=4, help='NSEW bounding box')

#I2M_master_p = subparsers.add_parser('I2M-master', help='Does I2M functions in the following order NADIR, nanrows ( keeps just those without NANrows), Derive band norms and BTDs removes original data ')
#I2M_master_p.set_defaults(func=I2M_master.main)
#I2M_master_p.add_argument('-q', '--quiet', action='count', default=0)
#I2M_master_p.add_argument('npz_path', help='Path to npz file')


######### MLR steps
#MLR_p = subparsers.add_parser('MLR-train', help='Train a MLR SKL model from .npz data and provide ML output')
#MLR_p.set_defaults(func=MLR_train.MLR)
#MLR_p.add_argument('npz_path', help='Path to npz file')
#MLR_p.add_argument('--DNB', help='the DNB truth we are using')
#MLR_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
#MLR_p.add_argument('-q', '--quiet', action='count', default=0)

######### FNN steps

FNN_prep_p = subparsers.add_parser('FNN-prep', help='Makes TORS/TANDS for dataset use (inclsues TAND_test/train TORS_test/train')
FNN_prep_p.set_defaults(func=FNN_prep.FNN_prep)
FNN_prep_p.add_argument('npz_path', help='Path to npz file')
FNN_prep_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
FNN_prep_p.add_argument('-q', '--quiet', action='count', default=0)

FNN_train_p = subparsers.add_parser('FNN-train', help='Train a FNN given a set of data')
FNN_train_p.set_defaults(func=FNN_train.FNN_train)
FNN_train_p.add_argument('npz_path', help='Path to npz file')
FNN_train_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
FNN_train_p.add_argument('-q', '--quiet', action='count', default=0)

FNN_GOES_expanded_train_p = subparsers.add_parser('FNN-GOES-train', help='Train a FNN given a set of data from GOES')
FNN_GOES_expanded_train_p.set_defaults(func=FNN_GOES_expanded_train.FNN_train)
FNN_GOES_expanded_train_p.add_argument('npz_path', help='Path to npz file')
FNN_GOES_expanded_train_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
FNN_GOES_expanded_train_p.add_argument('-q', '--quiet', action='count', default=0)

FNN_retrain_p = subparsers.add_parser('FNN-retrain', help='REtrain a FNN given a predictos/predictand array meant for use after inital train on data')
FNN_retrain_p.set_defaults(func=FNN_retrain.FNN_retrain)
FNN_retrain_p.add_argument('npz_path', help='Path to npz file of TORS/TAND')
FNN_retrain_p.add_argument('-q', '--quiet', action='count', default=0)

FNN_predict_p = subparsers.add_parser('FNN-predict', help='Predict ML values with FNN') #may become predict only) 
FNN_predict_p.set_defaults(func=FNN_predict.predict)
FNN_predict_p.add_argument('npz_path', help='Path to npz file with channels')
FNN_predict_p.add_argument('model_path', help='Path to Model folder')
FNN_predict_p.add_argument('channel_path', help='Path to Model Channels')
FNN_predict_p.add_argument('-q', '--quiet', action='count', default=0)        

FNN_GOES_predict_p = subparsers.add_parser('FNN-GOES-predict', help='Predict ML values with FNN - must adjust .py file for the "channels" in prgrm for C13,C14 or all and check -X save lenght') #may become predict only) 
FNN_GOES_predict_p.set_defaults(func=FNN_GOES_predict.predict)
FNN_GOES_predict_p.add_argument('npz_path', help='Path to npz file with channels')
FNN_GOES_predict_p.add_argument('model_path', help='Path to Model folder')
FNN_GOES_predict_p.add_argument('channel_path', help='Path to Model Channels')
FNN_GOES_predict_p.add_argument('-q', '--quiet', action='count', default=0)  

FNN_AHI_predict_p = subparsers.add_parser('FNN-AHI-predict', help='Predict ML values with FNN - must adjust .py file for the "channels" in prgrm for B13,B14 or B13 annd B14 model and check -X save lenght') #may become predict only) 
FNN_AHI_predict_p.set_defaults(func=FNN_AHI_predict.predict)
FNN_AHI_predict_p.add_argument('npz_path', help='Path to npz file with channels')
FNN_AHI_predict_p.add_argument('model_path', help='Path to Model folder')
FNN_AHI_predict_p.add_argument('channel_path', help='Path to Model Channels')
FNN_AHI_predict_p.add_argument('-q', '--quiet', action='count', default=0)  


FNN_assess_p = subparsers.add_parser('FNN-assess', help='compare ML vs DNB values') # may become train only) 
FNN_assess_p.set_defaults(func=FNN_assess.assessment)
FNN_assess_p.add_argument('npz_path', help='Path to npz file with normed values') #DNB and band raw values')
#FNN_assess_p.add_argument('norm_path', help='Path to npz file with normed values')
FNN_assess_p.add_argument('ML_path', help='Path to ML data')
FNN_assess_p.add_argument('-q', '--quiet', action='count', default=0)   
#removecloud_p = subparsers.add_parser('cloud', help='trying to remove clear')
#removecloud_p.set_defaults(func=removecloud.remove_cloud)
#removecloud_p.add_argument('npz_path', help='Path to npz file')
#removecloud_p.add_argument('--Predictors', nargs="+", help='the predictors we are using')
#removecloud_p.add_argument('-q', '--quiet', action='count', default=0)

#map_rawbands_p = subparsers.add_parser('map-rawband', help='images of the raw IRs for comparisons to Predict ML values') #may become predict only) 
#map_rawbands_p.set_defaults(func=map_rawbands.plot_raw)
#map_rawbands_p.add_argument('npz_path', help='Path to npz file with channels')
#map_rawbands_p.add_argument('-q', '--quiet', action='count', default=0) 


#map_MLNVI_p = subparsers.add_parser('map-MLNVI', help='images of the MLNVIs for comparisons to raw and SM reflectance values') #may become predict only) 
#map_MLNVI_p.set_defaults(func=map_MLNVI.plot_ML)
#map_MLNVI_p.add_argument('npz_path', help='Path to npz file with MLtruth')
#map_MLNVI_p.add_argument('-q', '--quiet', action='count', default=0)
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



