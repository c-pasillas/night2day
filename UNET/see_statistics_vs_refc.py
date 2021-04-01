import numpy as np
import sys

from custom_model_elements import my_r_square_metric
from prepare_data import ymin_default,ymax_default,xmin_default,xmax_default
from radar_reflec_cmap_clipped import cmap, norm, bounds, ticklabels
from scipy.stats import pearsonr

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

# get MRMS truth

Ydata_test = data['J']['Ydata_test']
ntest,ny,nx = Ydata_test.shape

# re-scale GOES predictions and MRMS

Ydata_test *= ( ymax_default - ymin_default)
Ydata_test += ymin_default

for apanel in panels:
    prediction[apanel] *= ( ymax_default - ymin_default )
    prediction[apanel] += ymin_default

# print statistics

rmsd = {}
rsq = {}
for apanel in panels:
    rmsd[apanel] = []
    rsq[apanel] = []

refs = [0,5,10,15,20,25,30,35,40,45,50]

print()
print('nrefs=',str(len(refs)))

print('numobs')
for iref,aref in enumerate(refs):

    bref = aref+5
    if aref == 50: bref = 75

    hit = (Ydata_test >= aref) & (Ydata_test < bref)

    print(iref, aref, np.sum(hit))

    for apanel in panels:

        rmsd[apanel].append( np.sqrt(np.mean( (prediction[apanel][hit]-Ydata_test[hit])**2 )) )
        rsq[apanel].append( pearsonr(prediction[apanel][hit],Ydata_test[hit])[0]**2 )

print('rmsd')
for iref,aref in enumerate(refs):
    print(iref, aref, \
        '{0:6.3f}'.format(rmsd['F'][iref]), \
        '{0:6.3f}'.format(rmsd['G'][iref]), \
        '{0:6.3f}'.format(rmsd['H'][iref]), \
        '{0:6.3f}'.format(rmsd['I'][iref]), \
        '{0:6.3f}'.format(rmsd['J'][iref]), \
        )

print('rsq')
for iref,aref in enumerate(refs):
    print(iref, aref, \
        '{0:6.4f}'.format(rsq['F'][iref]), \
        '{0:6.4f}'.format(rsq['G'][iref]), \
        '{0:6.4f}'.format(rsq['H'][iref]), \
        '{0:6.4f}'.format(rsq['I'][iref]), \
        '{0:6.4f}'.format(rsq['J'][iref]), \
        )

