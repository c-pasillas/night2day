from setup_performance_diagram import setup_performance_diagram
from matplotlib import pyplot as plt
import numpy as np

from custom_model_elements import my_r_square_metric, my_mean_squared_error_weighted1, my_mean_squared_error_weighted
from prepare_data import ymin_default,ymax_default,xmin_default,xmax_default
from refc_statistics import get_refc_stats

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    from tensorflow.keras import models
    from tensorflow.keras.models import load_model

data_file = '../DATA_SCALED/conus.npz'

#models = ['K5_0','K5_1','K5_2','K5_3','K5_4','K5_5']
#models = ['K5_2','K5_3','K5_L9']
#models = ['K5_2','K5_3','K5_G2.5']
models = ['K5_G2.5','K5_G2.0','K5_G1.5','K5_G1.0','K5_G0.5']

model_file = {}
model_file['K5_0'] = 'from_hera/model_K5_0_SEQ_blocks_3_epochs_25.h5'
model_file['K5_1'] = 'from_hera/model_K5_1_SEQ_blocks_3_epochs_25.h5'
model_file['K5_2'] = 'from_hera/model_K5_2_SEQ_blocks_3_epochs_25.h5'
model_file['K5_3'] = 'from_hera/model_K5_3_SEQ_blocks_3_epochs_25.h5'
model_file['K5_4'] = 'from_hera/model_K5_4_SEQ_blocks_3_epochs_25.h5'
model_file['K5_5'] = 'from_hera/model_K5_5_SEQ_blocks_3_epochs_25.h5'
model_file['K5_L9'] = 'from_hera/model_K5_L9_SEQ_blocks_3_epochs_25.h5'
model_file['K5_G2.5'] = 'from_hera/model_K5_G2.5_SEQ_blocks_3_epochs_25.h5'
model_file['K5_G2.0'] = 'from_hera/model_K5_G2.0_SEQ_blocks_3_epochs_25.h5'
model_file['K5_G1.5'] = 'from_hera/model_K5_G1.5_SEQ_blocks_3_epochs_25.h5'
model_file['K5_G1.0'] = 'from_hera/model_K5_G1.0_SEQ_blocks_3_epochs_25.h5'
model_file['K5_G0.5'] = 'from_hera/model_K5_G0.5_SEQ_blocks_3_epochs_25.h5'

colors = {}
colors['K5_0'] = 'black'
colors['K5_1'] = 'gray'
colors['K5_2'] = 'blue'
colors['K5_3'] = 'green'
colors['K5_4'] = 'orange'
colors['K5_5'] = 'red'
colors['K5_L9'] = 'red'
colors['K5_G2.5'] = 'red'
colors['K5_G2.0'] = 'orange'
colors['K5_G1.5'] = 'green'
colors['K5_G1.0'] = 'blue'
colors['K5_G0.5'] = 'black'

data = np.load(data_file)

model = {}
for amod in models:
    model[amod] = load_model(model_file[amod],\
        custom_objects={\
        "my_r_square_metric": my_r_square_metric,\
        "my_mean_squared_error_weighted1": my_mean_squared_error_weighted1,\
        "my_mean_squared_error_weighted": my_mean_squared_error_weighted,\
        "loss": my_mean_squared_error_weighted(),\
        })
# note: haven't loaded linear, gaussian custom objects - but shouldn't be necessary... just looking for 'loss'

prediction = {}
for amod in models:
    prediction[amod] = model[amod].predict(data['Xdata_test'])

Ydata_test = data['Ydata_test']
Ydata_test *= ( ymax_default - ymin_default)
Ydata_test += ymin_default

truth = Ydata_test
bad = truth < -99
truth = np.ma.masked_where(bad,truth)

for amod in models:
    prediction[amod] *= ( ymax_default - ymin_default )
    prediction[amod] += ymin_default

pstats = {}
fstats = {}
refs = [5,10,15,20,25,30,35,40,45,50]
labels = [str(aref) for aref in refs]
for amod in models:

    zz = prediction[amod]
    zz = np.ma.masked_where(bad,zz)

    astat = get_refc_stats(zz,truth,refs)

    pstats[amod] = np.array(astat['pod'])
    fstats[amod] = 1.0-np.array(astat['far'])

fig = plt.figure(figsize=(12,9))

setup_performance_diagram()

for amod in models:

    plt.plot(fstats[amod],pstats[amod],label=amod,color=colors[amod],linewidth=2)

    for iref in range(len(refs)):
        plt.text(fstats[amod][iref],pstats[amod][iref],labels[iref],color=colors[amod])

plt.legend()

#figname = '../OUTPUT/statistics_figures/fig_performance_diagram_K5.png'
#figname = '../OUTPUT/statistics_figures/fig_performance_diagram_K5_L9.png'
#figname = '../OUTPUT/statistics_figures/fig_performance_diagram_K5_G2.5.png'
figname = '../OUTPUT/statistics_figures/fig_performance_diagram_K5_G.png'
fig.savefig(figname,dpi=300)
print(figname)
