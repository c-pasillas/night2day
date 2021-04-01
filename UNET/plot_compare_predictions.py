import ir_radar
from ltg_radar import norm as ltg_norm
from ltg_radar import bounds as ltg_ticks

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys

from custom_model_elements import my_r_square_metric
from prepare_data import ymin_default,ymax_default,xmin_default,xmax_default
from radar_reflec_cmap_clipped import cmap, norm, bounds, ticklabels
from refc_statistics import get_refc_stats
from read_datetimes import read_datetimes

# custom loss functions
from custom_model_elements import my_mean_squared_error_noweight
from custom_model_elements import my_mean_squared_error_weighted1
from custom_model_elements import my_mean_squared_error_weighted

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    from tensorflow.keras import models
    from tensorflow.keras.models import load_model

# define data and models

panels = ['F','G','H','I','J']

data_file = {}
#
#data_file['F'] = '../DATA_SCALED/conusC13.npz'
#data_file['G'] = '../DATA_SCALED/conusC13GLM.npz'
#data_file['H'] = '../DATA_SCALED/conusC13C09.npz'
#data_file['I'] = '../DATA_SCALED/conusC13C09GLM.npz'
#data_file['J'] = '../DATA_SCALED/conus.npz'
#
#data_file['F'] = '../DATA_SCALED/conusC13.npz'
#data_file['G'] = '../DATA_SCALED/conusC13GLM.npz'
#data_file['H'] = '../DATA_SCALED/conusC13C09.npz'
#data_file['I'] = '../DATA_SCALED/conusC13C09.npz'
#data_file['J'] = '../DATA_SCALED/conusC13C09.npz'
#
data_file['F'] = '../DATA_SCALED/conus.npz'
data_file['G'] = '../DATA_SCALED/conus.npz'
data_file['H'] = '../DATA_SCALED/conus.npz'
data_file['I'] = '../DATA_SCALED/conus.npz'
data_file['J'] = '../DATA_SCALED/conus.npz'

model_file = {}
#
#model_file['F'] = '../OUTPUT/MODEL/model_K1_C13_SEQ_blocks_3_epochs_25.h5'
#model_file['G'] = '../OUTPUT/MODEL/model_K1_C13GLM_SEQ_blocks_3_epochs_25.h5'
#model_file['H'] = '../OUTPUT/MODEL/model_K1_C13C09_SEQ_blocks_3_epochs_25.h5'
#model_file['I'] = '../OUTPUT/MODEL/model_K1_C13C09GLM_SEQ_blocks_3_epochs_25.h5'
#model_file['J'] = '../OUTPUT/MODEL/model_K1_SEQ_blocks_3_epochs_25.h5'
#
#model_file['F'] = 'from_hera/model_K2_C13_SEQ_blocks_3_epochs_25.h5'
#model_file['G'] = 'from_hera/model_K2_C13GLM_SEQ_blocks_3_epochs_25.h5'
#model_file['H'] = 'from_hera/model_K2_C13C09_SEQ_blocks_3_epochs_25.h5'
#model_file['I'] = 'from_hera/model_K2_C13C09GLM_SEQ_blocks_3_epochs_25.h5'
#model_file['J'] = 'from_hera/model_K2_SEQ_blocks_3_epochs_25.h5'
#
#model_file['F'] = 'from_hera/model_K3_C13_SEQ_blocks_3_epochs_25.h5'
#model_file['G'] = 'from_hera/model_K3_C13GLM_SEQ_blocks_3_epochs_25.h5'
#model_file['H'] = 'from_hera/model_K3_C13C09_SEQ_blocks_3_epochs_25.h5'
#model_file['I'] = 'from_hera/model_K3_C13C09_SEQ_blocks_3_epochs_25.h5'
#model_file['J'] = 'from_hera/model_K3_C13C09_SEQ_blocks_3_epochs_25.h5'
#
#model_file['F'] = 'from_hera/model_K5_0_SEQ_blocks_3_epochs_25.h5'
#model_file['G'] = 'from_hera/model_K5_1_SEQ_blocks_3_epochs_25.h5'
#model_file['H'] = 'from_hera/model_K5_2_SEQ_blocks_3_epochs_25.h5'
#model_file['I'] = 'from_hera/model_K5_3_SEQ_blocks_3_epochs_25.h5'
#model_file['J'] = 'from_hera/model_K5_4_SEQ_blocks_3_epochs_25.h5'
#
model_file['F'] = 'from_hera/model_K5_0_SEQ_blocks_3_epochs_25.h5'
model_file['G'] = 'from_hera/model_K6_34_SEQ_blocks_3_epochs_25.h5'
model_file['H'] = 'from_hera/model_K6_40_SEQ_blocks_3_epochs_25.h5'
model_file['I'] = 'from_hera/model_K6_44_SEQ_blocks_3_epochs_25.h5'
model_file['J'] = 'from_hera/model_K6_39_SEQ_blocks_3_epochs_25.h5'

model_name = {}
#
#model_name['F'] = 'C13'
#model_name['G'] = 'C13+GLM'
#model_name['H'] = 'C13+C09'
#model_name['I'] = 'C13+C09+GLM'
#model_name['J'] = 'C13+C09+C07+GLM'
#
#model_name['F'] = 'C13'
#model_name['G'] = 'C13+GLM'
#model_name['H'] = 'C13+C09'
#model_name['I'] = 'C13+C09'
#model_name['J'] = 'C13+C09'
#
#model_name['F'] = 'Wt=0'
#model_name['G'] = 'Wt=1'
#model_name['H'] = 'Wt=2'
#model_name['I'] = 'Wt=3'
#model_name['J'] = 'Wt=4'
#
model_name['F'] = 'K5_0'
model_name['G'] = 'K6_34'
model_name['H'] = 'K6_40'
model_name['I'] = 'K6_44'
model_name['J'] = 'K6_39'

# load data and models, make predictions

data = {}
for apanel in panels:
    data[apanel] = np.load(data_file[apanel])

model = {}
for apanel in panels:
    model[apanel] = load_model(model_file[apanel],\
        custom_objects = { \
        "my_r_square_metric": my_r_square_metric,\
        "my_mean_squared_error_weighted1": my_mean_squared_error_weighted1,\
        "my_mean_squared_error_weighted": my_mean_squared_error_weighted,\
        "loss": my_mean_squared_error_weighted(),\
        })

prediction = {}
for apanel in panels:
    prediction[apanel] = model[apanel].predict(data[apanel]['Xdata_test'])

datetimes = read_datetimes('samples_datetimes.txt')

# get GOES, MRMS inputs

Lat_test = data['J']['Lat_test']
Lon_test = data['J']['Lon_test']
Xdata_test = data['J']['Xdata_test']
Ydata_test = data['J']['Ydata_test']
ntest,ny,nx,nchan = Xdata_test.shape
zfill = int(np.ceil(np.log10(ntest)))

# re-scale GOES, MRMS, and predictions

channels = {0:7, 1:9, 2:13, 3:'GROUP'}

for ichan in [0,1,2,3]:
    achan = channels[ichan]
    if achan == 'GROUP':
#        Xdata_test[:,:,:,ichan] *= ( xmax_default[achan] - xmin_default[achan] )
#        Xdata_test[:,:,:,ichan] += xmin_default[achan]
        Xdata_test[:,:,:,ichan] *= xmax_default[achan]
        Xdata_test[:,:,:,ichan][Xdata_test[:,:,:,ichan] < xmin_default[achan]] = 0.
    elif achan >= 7:
        Xdata_test[:,:,:,ichan] *= -1*( xmax_default[achan] - xmin_default[achan] )
        Xdata_test[:,:,:,ichan] += xmax_default[achan]
    else:
        sys.exit('error')

Ydata_test *= ( ymax_default - ymin_default)
Ydata_test += ymin_default

for apanel in panels:
    prediction[apanel] *= ( ymax_default - ymin_default )
    prediction[apanel] += ymin_default

# plot figures

fig = plt.figure(figsize=(6.5,3.5))
plt.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.95,wspace=0.3,hspace=0.6)
ncol = 5
nrow = 2
fs = 'xx-small'

#for itest in range(ntest):
for itest in [68]:

    datestring = datetimes['TEST'][itest].strftime('%Y%m%d%H%MZ')

    lat = Lat_test[itest,:,:]
    lon = Lon_test[itest,:,:]

    basemap = {}
    basemap['projection'] = 'cyl'
    basemap['resolution'] = 'l'
    basemap['llcrnrlon'] = np.min(lon)
    basemap['urcrnrlon'] = np.max(lon)
    basemap['llcrnrlat'] = np.min(lat)
    basemap['urcrnrlat'] = np.max(lat)
    basemap['fix_aspect'] = False
    basemap = Basemap(**basemap)
    xb,yb = basemap(lon,lat)

    bad = Ydata_test[itest,:,:] < -99

    tbcmap = plt.get_cmap('ir')
    vmin_tb = 200
    vmax_tb = 280

    plt.subplot(nrow,ncol,1)
    zz = Xdata_test[itest,:,:,0]
    zz = np.ma.masked_where(bad,zz)
    pcm = basemap.pcolormesh(xb,yb,zz,cmap=tbcmap,vmin=vmin_tb,vmax=vmax_tb)
    basemap.drawcoastlines()
    basemap.drawcountries()
    basemap.drawstates()
    basemap.drawcounties()
    plt.title('(A) GOES-16 C07',fontsize=fs)

    plt.subplot(nrow,ncol,2)
    zz = Xdata_test[itest,:,:,1]
    zz = np.ma.masked_where(bad,zz)
    pcm = basemap.pcolormesh(xb,yb,zz,cmap=tbcmap,vmin=vmin_tb,vmax=vmax_tb)
    basemap.drawcoastlines()
    basemap.drawcountries()
    basemap.drawstates()
    basemap.drawcounties()
    plt.title('(B) GOES-16 C09',fontsize=fs)

    plt.subplot(nrow,ncol,3)
    zz = Xdata_test[itest,:,:,2]
    zz = np.ma.masked_where(bad,zz)
    pcm = basemap.pcolormesh(xb,yb,zz,cmap=tbcmap,vmin=vmin_tb,vmax=vmax_tb)
    basemap.drawcoastlines()
    basemap.drawcountries()
    basemap.drawstates()
    basemap.drawcounties()
    plt.title('(C) GOES-16 C13',fontsize=fs)

    cbax = fig.add_axes([0.03, 0.56, 0.54, 0.015])
    cb = plt.colorbar(pcm,orientation='horizontal',cax=cbax)
    cb.set_label('Brightness Temperature (K)',fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    plt.subplot(nrow,ncol,4)
    zz = Xdata_test[itest,:,:,3]
    zz = np.ma.masked_where(bad,zz)
    pcm = basemap.pcolormesh(xb,yb,zz,cmap='ltg',norm=ltg_norm)
    basemap.drawcoastlines()
    basemap.drawcountries()
    basemap.drawstates()
    basemap.drawcounties()
    plt.title('(D) GOES-16 GLM Groups',fontsize=fs)

    cbax = fig.add_axes([0.63, 0.56, 0.14, 0.015])
    cb = plt.colorbar(pcm,orientation='horizontal',cax=cbax,ticks=ltg_ticks[::2])
    cb.set_label('Groups/5-minutes/km^2',fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    plt.subplot(nrow,ncol,5)
    zz = Ydata_test[itest,:,:]
    zz = np.ma.masked_where(bad,zz)
    pcm = basemap.pcolormesh(xb,yb,zz,cmap=cmap,norm=norm)
    basemap.drawcoastlines()
    basemap.drawcountries()
    basemap.drawstates()
    basemap.drawcounties()
    plt.title('(E) MRMS REFC',fontsize=fs)

    cbax = fig.add_axes([0.83, 0.56, 0.14, 0.015])
    cb = plt.colorbar(pcm,orientation='horizontal',cax=cbax)
    cb.ax.set_xticklabels(ticklabels)
    cb.set_label('Composite Reflectivity (dBZ)',fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    cb.ax.set_xticklabels([0,10,20,30,40,50,60])

    for apanel,ipanel in zip(panels,[6,7,8,9,10]):
        plt.subplot(nrow,ncol,ipanel)
        zz = prediction[apanel][itest,:,:]
        zz = np.ma.masked_where(bad,zz)
        pcm = basemap.pcolormesh(xb,yb,zz,cmap=cmap,norm=norm)
        basemap.drawcoastlines()
        basemap.drawcountries()
        basemap.drawstates()
        basemap.drawcounties()
        stats = get_refc_stats(zz,Ydata_test[itest,:,:],[20,35])
        items = []
        items.append('{0:3.1f}'.format(stats['rmsd']))
        items.append('{0:3.2f}'.format(stats['rsq']))
        items.append('{0:2.0f}'.format(np.max(zz)))
        items.append('{0:3.2f}'.format(stats['csi'][0]))
        items.append('{0:3.2f}'.format(stats['csi'][1]))
        stat_string = ', '.join(items)
        plt.title('('+apanel+') '+ model_name[apanel] + '\n' + stat_string,fontsize=fs)


#    figname = '../OUTPUT/Prediction_Figures/fig_'+str(itest).zfill(zfill)+'_'+datestring+'.png'
#    figname = '../OUTPUT/Prediction_Figures_WtLoss/fig_'+str(itest).zfill(zfill)+'_'+datestring+'.png'
#    figname = '../OUTPUT/Prediction_Figures_1x1/fig_'+str(itest).zfill(zfill)+'_'+datestring+'.png'
#    figname = '../OUTPUT/Prediction_Figures_Wt/fig_'+str(itest).zfill(zfill)+'_'+datestring+'.png'
    figname = '../OUTPUT/Prediction_Figures_Wt_K6/fig_'+str(itest).zfill(zfill)+'_'+datestring+'.png'

    fig.savefig(figname,dpi=300)
    print(figname)
    plt.clf()

#    sys.exit('stop here')
