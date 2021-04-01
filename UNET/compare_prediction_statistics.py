import numpy as np
import sys

from custom_model_elements import my_r_square_metric
from prepare_data import ymin_default,ymax_default,xmin_default,xmax_default
from radar_reflec_cmap_clipped import cmap, norm, bounds, ticklabels
from refc_statistics import get_refc_stats
from read_datetimes import read_datetimes

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    from tensorflow.keras import models
    from tensorflow.keras.models import load_model

# define data and models

panels = ['F','G','H','I','J']

data_file = {}
data_file['F'] = '../DATA_SCALED/conusC13.npz'
data_file['G'] = '../DATA_SCALED/conusC13GLM.npz'
data_file['H'] = '../DATA_SCALED/conusC13C09.npz'
data_file['I'] = '../DATA_SCALED/conusC13C09GLM.npz'
data_file['J'] = '../DATA_SCALED/conus.npz'

model_file = {}
model_file['F'] = '../OUTPUT/MODEL/model_K1_C13_SEQ_blocks_3_epochs_25.h5'
model_file['G'] = '../OUTPUT/MODEL/model_K1_C13GLM_SEQ_blocks_3_epochs_25.h5'
model_file['H'] = '../OUTPUT/MODEL/model_K1_C13C09_SEQ_blocks_3_epochs_25.h5'
model_file['I'] = '../OUTPUT/MODEL/model_K1_C13C09GLM_SEQ_blocks_3_epochs_25.h5'
model_file['J'] = '../OUTPUT/MODEL/model_K1_SEQ_blocks_3_epochs_25.h5'

model_name = {}
model_name['F'] = 'C13'
model_name['G'] = 'C13+GLM'
model_name['H'] = 'C13+C09'
model_name['I'] = 'C13+C09+GLM'
model_name['J'] = 'C13+C09+C07+GLM'

# load data and models, make predictions

data = {}
for apanel in panels:
    data[apanel] = np.load(data_file[apanel])

model = {}
for apanel in panels:
    model[apanel] = load_model(model_file[apanel],\
        custom_objects={"my_r_square_metric": my_r_square_metric})

prediction = {}
for apanel in panels:
    prediction[apanel] = model[apanel].predict(data[apanel]['Xdata_test'])

datetimes = read_datetimes('samples_datetimes.txt')

# get GOES, MRMS inputs

Xdata_test = data['J']['Xdata_test']
Ydata_test = data['J']['Ydata_test']
ntest,ny,nx,nchan = Xdata_test.shape
zfill = int(np.ceil(np.log10(ntest)))

# re-scale GOES, MRMS, and predictions

channels = {0:7, 1:9, 2:13, 3:'GROUP'}

Ydata_test *= ( ymax_default - ymin_default)
Ydata_test += ymin_default

for apanel in panels:
    prediction[apanel] *= ( ymax_default - ymin_default )
    prediction[apanel] += ymin_default

# print statistics

for itest in range(ntest):

    datestring = datetimes['TEST'][itest].strftime('%Y%m%d%H%MZ')

    truth = Ydata_test[itest,:,:]
    bad = truth < -99
    truth = np.ma.masked_where(bad,truth)

    aline = str(itest).zfill(zfill) + ' ' + datestring

    for apanel in panels:

        zz = prediction[apanel][itest,:,:]
        zz = np.ma.masked_where(bad,zz)

        stats = get_refc_stats(zz,truth,[20,35])
        items = []
        items.append('{0:6.3f}'.format(stats['rmsd']))
        items.append('{0:6.4f}'.format(stats['rsq']))
        items.append('{0:3.1f}'.format(np.max(zz)))
        items.append('{0:5.3f}'.format(stats['csi'][0]))
        items.append('{0:5.3f}'.format(stats['csi'][1]))
        items.append('{0:5.3f}'.format(stats['pod'][0]).strip()) #some are nan
        items.append('{0:5.3f}'.format(stats['pod'][1]).strip())
        items.append('{0:5.3f}'.format(stats['far'][0]).strip())
        items.append('{0:5.3f}'.format(stats['far'][1]).strip())
        stat_string = ','.join(items)
        aline += stat_string

    print(aline)

#    sys.exit()
