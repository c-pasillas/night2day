import numpy as np

# Read data and pre-process (normalization, etc.)
from prepare_data import prepare_data
from make_custom_file_names import data_file_name

data_file = data_file_name()

filename_format = '../SAMPLES/case{0:02n}.nc'
ncases = 92
ncases_test = 18
cases_train = [i+1 for i in range(ncases-ncases_test)]
cases_test = [i+ncases-ncases_test+1 for i in range(ncases_test)]
channels = ['C07','C09','C13','GROUP']
qctimes_train = [] #all good
qctimes_test = [] #all good

train = prepare_data(filename_format, cases_train, channels, qctimes_train)
test = prepare_data(filename_format, cases_test, channels, qctimes_test)

print('*TRAIN*')
for i in range(train['nsamples']):
    print(i,train['Dates'][i].strftime('%Y%m%d%H%MZ'))

print('*TEST*')
for i in range(test['nsamples']):
    print(i,test['Dates'][i].strftime('%Y%m%d%H%MZ'))
