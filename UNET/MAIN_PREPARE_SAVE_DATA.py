import numpy as np
import sys
from datetime import datetime

# Read data and pre-process (normalization, etc.)
from prepare_data import prepare_data
from make_custom_file_names import data_file_name
from read_configuration import read_configuration
from default_configuration import defcon
from smooth import smooth

print('start MAIN_PREPARE_SAVE_DATA=',datetime.now())

# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()

try:
    machine = config['machine']
except KeyError:
    try:
        machine = config[my_file_prefix]['machine']
    except KeyError:
        machine = defcon['machine']
print('machine =',machine)

spath = '..'  #Imme
if machine == 'Hera':
    spath = '/scratch1/RDARCH/rda-goesstf/conus2'  #KH on Hera

try:
    data_suffix = config['data_suffix']
except KeyError:
    try:
        data_suffix = config[my_file_prefix]['data_suffix']
    except KeyError:
        data_suffix = defcon['data_suffix']
print('data_suffix =',data_suffix)

data_file = data_file_name( spath, suffix=data_suffix ) # get data file name

if data_suffix in ['vis','visa','visb','visc','vis_scalar']:
    filename_format = spath+'/SAMPLES_VIS/case{0:02n}.nc'
if data_suffix in ['vis2']:
    filename_format = spath+'/SAMPLES_VIS2/case{0:02n}.nc'
elif data_suffix in ['ctc']:
    filename_format = spath+'/SAMPLES_CTC/case{0:02n}.nc'
elif data_suffix in ['ctc2']:
    filename_format = spath+'/SAMPLES_CTC2/case{0:02n}.nc'
elif data_suffix in ['uv']:
    filename_format = spath+'/SAMPLES_UV/case{0:02n}.nc'
else:
    filename_format = spath+'/SAMPLES/case{0:02n}.nc'

ncases = 92
ncases_test = 18
cases_train = [i+1 for i in range(ncases-ncases_test)]
cases_test = [i+ncases-ncases_test+1 for i in range(ncases_test)]
channels = ['C07','C09','C13','GROUP']
qctimes_train = [] #all good
qctimes_test = [] #all good
nscalar = 0  #initialize

if data_suffix == 'LTG1':
    xmin = {1:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.0}
    xmax = {1:1.00, 6:1.00, 7:300, 9:300, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
#elif data_suffix == 'rescaleC09':
elif data_suffix in ['rescaleC09','C13C09rescale','C13C09GLMrescale']:
    xmin = {1:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {1:1.00, 6:1.00, 7:300, 9:250, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
elif data_suffix == 'met':
    channels = ['C07','C09','C13','GROUP','HRRR_H500','HRRR_THICK','HRRR_TPW']
    xmin = {1:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1,'HRRR_H500':5400,'HRRR_THICK':5350,'HRRR_TPW':0}
    xmax = {1:1.00, 6:1.00, 7:300, 9:250, 13:300, 'GROUP':50.0,'HRRR_H500':6000,'HRRR_THICK':5950,'HRRR_TPW':80}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
elif data_suffix == 'vis':
    channels = ['C02','C06','C07','C09','C13','GROUP']
    xmin = {2:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {2:1.00, 6:0.60, 7:300, 9:250, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
    print('number of samples train/test=',train['nsamples'],test['nsamples'])
elif data_suffix == 'visa':
    qctimes_train = [datetime(2019,4,25,4,45),datetime(2019,4,24,3,45)]
    qctimes_test = [datetime(2019,7,9,1,45)]
    channels = ['C02','C06','C07','C09','C13','GROUP']
    xmin = {2:0.01, 6:0.01, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {2:1.00, 6:0.60, 7:300, 9:250, 13:300, 'GROUP':50.0}
    print('qctimes_train=',qctimes_train)
    print('qctimes_test=',qctimes_test)
    print('channels=',channels)
    print('xmin=',xmin)
    print('xmax=',xmax)
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
    print('number of samples train/test=',train['nsamples'],test['nsamples'])
elif data_suffix == 'visb':
    qctimes_train = [datetime(2019,4,25,4,45),datetime(2019,4,24,3,45)]
    qctimes_test = [datetime(2019,7,9,1,45)]
    channels = ['C02','C06','C07','C09','C13','GROUP','SZA']
    xmin = {2:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {2:1.00, 6:0.60, 7:300, 9:250, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
elif data_suffix == 'visc':
    qctimes_train = [datetime(2019,4,25,4,45),datetime(2019,4,24,3,45)]
    qctimes_test = [datetime(2019,7,9,1,45)]
    channels = ['C02','C06','C07','C09','C13','GROUP','SZA']
    xmin = {2:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {2:1.50, 6:0.90, 7:300, 9:250, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax,szanorm=True)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax,szanorm=True)
elif data_suffix == 'vis_scalar':
    nscalar = 1
    iscalar = 6
    qctimes_train = [datetime(2019,4,25,4,45),datetime(2019,4,24,3,45)]
    qctimes_test = [datetime(2019,7,9,1,45)]
    channels = ['C02','C06','C07','C09','C13','GROUP','SZA']
    xmin = {2:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {2:1.00, 6:0.60, 7:300, 9:250, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
elif data_suffix == 'vis2':
    qctimes_train = [datetime(2019,4,25,4,45),datetime(2019,4,24,3,45)]
    qctimes_test = [datetime(2019,7,9,1,45)]
    channels = ['C02','C06','C07','C09','C13','GROUP','SZA','SGA']
    xmin = {2:0.00, 6:0.00, 7:200, 9:200, 13:200, 'GROUP':0.1}
    xmax = {2:1.00, 6:0.60, 7:300, 9:250, 13:300, 'GROUP':50.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
elif data_suffix in ['ctc','ctc2']:
    channels = ['C07','C09','C13','GROUP','CTC']
    xmin = {7:200, 9:200, 13:200, 'GROUP':0.1, 'CTC':2.0}
    xmax = {7:300, 9:250, 13:300, 'GROUP':50.0, 'CTC':20.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
elif data_suffix in ['uv']:
    channels = ['C07','C09','C13','GROUP','U','V']
    xmin = {7:200, 9:200, 13:200, 'GROUP':0.1, 'U':-2.0, 'V':-2.0}
    xmax = {7:300, 9:250, 13:300, 'GROUP':50.0, 'U':2.0, 'V':2.0}
    train = prepare_data(filename_format, cases_train, channels, qctimes_train, xmin=xmin,xmax=xmax)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test, xmin=xmin,xmax=xmax)
else:
    train = prepare_data(filename_format, cases_train, channels, qctimes_train)
    test = prepare_data(filename_format, cases_test, channels, qctimes_test)

if data_suffix in ['ctc','ctc2']:
    for ibatch in range(train['nbatches']):
        train['Xdata'][ibatch,:,:,4] = smooth(\
            train['Xdata'][ibatch,:,:,4],\
            np.isfinite(train['Xdata'][ibatch,:,:,4]), (3,3))
    for ibatch in range(test['nbatches']):
        test['Xdata'][ibatch,:,:,4] = smooth(\
            test['Xdata'][ibatch,:,:,4],\
            np.isfinite(test['Xdata'][ibatch,:,:,4]), (3,3))

if data_suffix == 'C13':
    print('zero out C07, C09, GLM')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,1] = 0.  #zero out C09
    train['Xdata'][:,:,:,3] = 0.  #zero out GLM
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,1] = 0.  #zero out C09
    test['Xdata'][:,:,:,3] = 0.  #zero out GLM

if data_suffix == 'C13GLM':
    print('zero out C07, C09')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,1] = 0.  #zero out C09
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,1] = 0.  #zero out C09

#if data_suffix == 'C13C09':
if data_suffix in ['C13C09','C13C09rescale']:
    print('zero out C07, GLM')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,3] = 0.  #zero out GLM
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,3] = 0.  #zero out GLM

#if data_suffix == 'C13C09GLM':
if data_suffix in ['C13C09GLM','C13C09GLMrescale']:
    print('zero out C07')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,0] = 0.  #zero out C07

if data_suffix == 'C09':
    print('zero out C07, C13, GLM')
    train['Xdata'][:,:,:,0] = 0.  #zero out C07
    train['Xdata'][:,:,:,2] = 0.  #zero out C13
    train['Xdata'][:,:,:,3] = 0.  #zero out GLM
    test['Xdata'][:,:,:,0] = 0.  #zero out C07
    test['Xdata'][:,:,:,2] = 0.  #zero out C13
    test['Xdata'][:,:,:,3] = 0.  #zero out GLM

print('number of samples train/test=',train['nsamples'],test['nsamples'])

print('Saving data to file:' + data_file)
if nscalar == 0:
    print('no scalars')
    np.savez( data_file, Xdata_train=train['Xdata'], Ydata_train=train['Ydata'],
       Xdata_test=test['Xdata'], Ydata_test=test['Ydata'],
       Lat_train=train['Lat'], Lon_train=train['Lon'],
       Lat_test=test['Lat'], Lon_test=test['Lon'] )
else:
    print('nscalar=',nscalar)
    np.savez( data_file, \
        Xdata_train        = train['Xdata'][:,:,:,0:iscalar], \
        Xdata_test         =  test['Xdata'][:,:,:,0:iscalar], \
        Xdata_scalar_train = train['Xdata'][:,:,:,iscalar:iscalar+1], \
        Xdata_scalar_test  =  test['Xdata'][:,:,:,iscalar:iscalar+1], \
        Ydata_train        = train['Ydata'], \
        Ydata_test         =  test['Ydata'], \
        Lat_train          = train['Lat'], \
        Lat_test           =  test['Lat'], \
        Lon_train          = train['Lon'], \
        Lon_test           =  test['Lon'] )

print('end MAIN_PREPARE_SAVE_DATA=',datetime.now())
