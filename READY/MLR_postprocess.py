#!/usr/bin/env python
# coding: utf-8


#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
from sklearn import linear_model
from sklearn import metrics
import os
import sys
import seaborn as sb
from matplotlib import style
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
scaler = StandardScaler()
from common import log
from pathlib import Path
from datetime import datetime
import normalize
from PIL import Image

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
dpiFig = 300.

style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (20,10)

#determine what we care about
predict_channels = ['M12norm','M13norm','M14norm','M15norm','M16norm','BTD1215norm','BTD1216norm','BTD1315norm',
'BTD1316norm','BTD1415norm','BTD1416norm','BTD1516norm']

#predictand_channels = [ 'DNB_log_Miller_full_moon']
#predictand_channels = ['DNB_log_norm']#,'DNB_log_full_moon_norm', 'DNB_log_new_moon_norm','DNB_log_Miller_full_moon']

predictand_channels = ['DNB_norm', 'DNB_full_moon_norm', 'DNB_new_moon_norm', 'DNB_log_norm','DNB_log_full_moon_norm', 'DNB_log_new_moon_norm','DNB_log_Miller_full_moon']

#%%
#helper function for plotting histogram and PDFs

def plotdata(truth, MLR, channel_name,figdir, nick):
    Y1 = scaler.fit_transform(truth.reshape(-1,1))
    Y2 = scaler.fit_transform(MLR.reshape(-1,1))

    #plotting basics
    # set figure defaults
    mpl.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = (10.0/2, 8.0/2)
    xinc = 0.1
    xbins = np.arange(-4, 4.2, xinc)

    #histogram counts
    plt.figure()
    hY1 = np.histogram(Y1,xbins)
    hY2 = np.histogram(Y2,xbins)

    plt.xlabel('value')
    plt.ylabel('counts')

    plt.bar(hY1[1][:-1],hY1[0],edgecolor = 'r', color = [], width = .2, label = 'truth', linewidth = 2)
    plt.bar(hY2[1][:-1],hY2[0],edgecolor = 'b', color = [], width = .2, label = 'MLR', linewidth = 2)
   
    plt.legend()
    plt.title(channel_name + ' by count')
    plt.savefig(figdir / f"{nick}_{channel_name}_MLR_histogram.png")
    #plt.show()
    plt.close()
    
    #make as PDF value 1
    plt.figure()
    xvals = hY1[1][:-1]
    fvalsY1 = hY1[0].astype(float)/(np.size(Y1)*xinc)
    fvalsY2 = hY2[0].astype(float)/(np.size(Y2)*xinc)

    plt.plot(xvals+xinc/2,fvalsY1,'r', label = 'truth')
    plt.plot(xvals+xinc/2,fvalsY2,'b', label = 'MLR')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title(channel_name + ' PDFs')
    plt.legend()
    plt.savefig(figdir / f"{nick}_{channel_name}_MLR_PDF.png")
    #plt.show()
    plt.close()
    
#helper save co stuff
def save_co_tables(table,filename):
    with open(filename,"w") as f:
        print(table, file =f)

def ensure_outputdir(npz_filename, nick):
    datasources = Path(npz_filename).resolve().parent.name
    f= Path(npz_filename).resolve().parent / 'MLR' / f'{datasources}_{nick}'
    f.mkdir(exist_ok=True, parents=True)
    return f      

#helpers of the helper channel
                 
def ERF(arr, label):
     # take the radiances and use the ERF display image scale
    log.info(f'there are this many NANS {np.sum(np.isnan(arr))}')
    Rmax= 1.26e-10 #check right constants
    Rmin=2e-11                  
    BVI = 255 * np.sqrt(np.abs((arr - Rmin)/(Rmax - Rmin)))                 
    log.info(f"ERF {label} max/min {BVI.max()},{BVI.min()}")
    return BVI    

def show_byte_img(arr, name='temp.png', **args):
    clip_lo, clip_hi = np.array([np.sum(arr < 0), np.sum(arr > 256)]) * 100 / arr.size
    print(f'clipping low/high: {clip_lo:.1f}% {clip_hi:.1f}%')
    b = arr.clip(min=0, max=255).astype('uint8')
    Image.fromarray(b).save(name)
    #dis.display(dis.Image(name, **args))

def standardize(arr, mean=150, std_dev=50, invert=1):
    norm = (arr - arr.mean()) / arr.std()
    return (norm * invert * std_dev) + mean

def scale(arr, percent_tail=2, percent_top=None, invert=False):
    left, right = (percent_tail/2, percent_tail/2) if percent_top is None \
                  else (percent_tail, percent_top)
    norm = (arr - arr.mean()) / arr.std()
    normi = norm * (1 if not invert else -1)
    sort_arr = np.sort(normi.flatten())
    left, right = int(sort_arr.size * left / 100), int(sort_arr.size * right / 100)
    print(f'left={left} right={right}')
    lo, hi = sort_arr[left], sort_arr[-(1 + right)]
    byte_scale = 256 / (hi - lo)
    offset = 0 - lo * byte_scale
    print(f'byte_scale={byte_scale:.2f} offset={offset:.2f}')
    return (normi * byte_scale) + offset


#helper one channel
def process_channel(Ycol, MLRcol, c, figdir, nick, metdict, shape, denormed):
    R2 = metrics.r2_score(Ycol, MLRcol)
    RMSE = metrics.mean_squared_error(Ycol, MLRcol, squared =False)
    MAE = metrics.mean_absolute_error(Ycol, MLRcol)
    metdict[c]= {}                   
    metdict[c]["regressions"] = f'R2 = {R2:.4f}, RMSE = {RMSE:.4f}, MAE = {MAE:.4f}'
    log.info(f'{c} y variance explained by all predictors fit = ' + str(np.round(R2,5)))
    log.info(f'{c} RMSE = ' + str(np.round(RMSE,5)))
    log.info(f'{c} MAE = ' + str(np.round(MAE,5)))
    plotdata(Ycol, MLRcol, c,figdir, nick)

    MLR_pred = normalize.denormalize(c, MLRcol)
    truth = normalize.denormalize(c, Ycol)
    MLR_pred_r = MLR_pred.reshape(shape)
    truth_r = truth.reshape(shape)
    
    denormed[c] = np.stack((MLR_pred_r, truth_r ), axis = -1)
    
    #boxplots
    log.info(' making boxplots')
    plt.boxplot([truth.flatten(), MLR_pred.flatten()], labels=["truth", "MLR"], sym='')
    plt.title(f'Data Point Distribution for {c}')
    plt.savefig(figdir / f"{nick}_{c}_boxplot.png")
    plt.close()

    plt.boxplot(truth.flatten(), sym='')
    plt.title(f'Data Point Distribution for {c} truth')
    plt.savefig(figdir / f"{nick}_{c}_boxplot_truth.png")
    plt.close()

    plt.boxplot(MLR_pred.flatten(), sym='')
    plt.title(f'Data Point Distribution for {c} MLR')
    plt.savefig(figdir / f"{nick}_{c}_boxplot_MLR.png")
    plt.close()
    
    coledir = figdir / "RAWimages"
    coledir.mkdir(exist_ok=True, parents=True)
    for i in range(shape[0]):
        #COLE PLOTTING
        show_byte_img(scale(truth_r[i], 5), name= coledir / f"{nick}_{c}_COLE_truth_{i}.png")
        show_byte_img(scale(MLR_pred_r[i], 5), name= coledir / f"{nick}_{c}_COLE_MLR_truth_{i}.png")
        
        
    truth_min, truth_max = np.nanmin(truth), np.nanmax(truth)
    MLR_min, MLR_max = np.nanmin(MLR_pred), np.nanmax(MLR_pred)                                 
    metdict[c]["denorm_raw"] = f'truth min/max = {truth_min:.4f}/{truth_max:.4f}, MLR_truth min/max = {MLR_min:.4f}/{MLR_max:.4f}'                   
    plotdata(MLR_pred, truth, c, figdir, nick + "denorm")
     
    #ERF applications on ML and truth values for image translation
    log.info(f'starting the ERF processes')
    ERFimage_truth, ERFimage_ML = ERF(truth, "truth"), ERF(MLR_pred, "MLR_pred") 
         
    # plotting image pair
    x=ERFimage_truth.reshape(shape)#[:2000]
    y=ERFimage_ML.reshape(shape)#[:2000]

    denormed[f'{c}_BVI'] = np.stack((y, x), axis=-1)

    imagedir = figdir / "ERFimages"
    imagedir.mkdir(exist_ok=True, parents=True)
    
    # makes ERF 
    for i in range(shape[0]):
        log.info(f'processing {i} image plotting')
        #ERF PLOTTING
        img = x[i]
        imgplot = plt.imshow(img, cmap='gray', vmin=2000, vmax=5000)
        plt.grid (False)
        plt.title('ERF imagery truth')
        plt.colorbar()
        plt.savefig(imagedir / f"{nick}_{c}_ERFtruth_{i}.png")
        #plt.show()
        plt.close()

        img2 = y[i]
        imgplot= plt.imshow(img2, cmap='gray',vmin=2000, vmax=5000)
        plt.grid (False)
        plt.title("ERF imagery ML")
        plt.colorbar()
        plt.savefig(imagedir / f"{nick}_{c}_ERF_MLR_truth_{i}.png")
        #plt.show()
        plt.close()
  
    
#MLR 

def postprocess (npzfilename, modelfilename, nick) :
    log.info("starting MLR postprocessing")
    figdir=ensure_outputdir(npzfilename, nick)
    f = np.load(npzfilename)
    shape = f['DNB_norm'].shape
    #make predictor/predictand arraysand flatten for use
    X = np.stack([f[key].flatten() for key in predict_channels], axis = -1)
    Y = np.stack([f[key].flatten() for key in predictand_channels], axis = -1)
    
    with open (modelfilename, 'rb') as g:
        model = pickle.load(g)
   #get the R2 
    MLR_truths =model.predict(X)
    
    metdict= {}
    
    denormed = {}
    
    for i,c in enumerate(predictand_channels):
        Ycol = Y[:,i]
        MLRcol = MLR_truths[:,i]
        process_channel(Ycol, MLRcol, c, figdir, nick, metdict, shape, denormed)
                     
    p= predictand_channels[0] if len(predictand_channels) == 1 else 'ALL'
    
    np.savez(figdir / f'{nick}_MLR_{p}_denormed_true_pred.npz', **denormed)
    
    with open (figdir / f'{nick}_MLR_{p}_postprocess_eventlog.txt', 'w') as f:
        x=Path(npzfilename).resolve().parent.name     
        print(datetime.now(), file = f)
        print(x, file = f)
        print(nick, file = f)
        print(metdict, file = f)
           
    
                 

   




    

