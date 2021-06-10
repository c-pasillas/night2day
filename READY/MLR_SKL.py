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
    plt.savefig(figdir / f"fitted_{channel_name}_MLR_histogram.png")
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
    plt.savefig(figdir / f"fitted_{channel_name}_MLR_PDF.png")
    #plt.show()
    plt.close()
    
#helper save co stuff
def save_co_tables(table,filename):
    with open(filename,"w") as f:
        print(table, file =f)

def ensure_figdir(npz_filename, nick):
    f= Path(npz_filename).resolve().parent / 'SKL' / nick
    f.mkdir(exist_ok=True, parents=True)
    return f      
    
#%%
#MLR for numpy

def MLR (filename, nick):
    log.info("starting MLR")
    figdir=ensure_figdir(filename, nick)
    f = np.load(filename)
    #make predictor/predictand arraysand flatten for use
    X = np.stack([f[key].flatten() for key in predict_channels], axis = -1)
    Y = np.stack([f[key].flatten() for key in predictand_channels], axis = -1)
    log.info("fitting MLR")
    regOLS = linear_model.LinearRegression(normalize=False, fit_intercept=True)   
    regOLS.fit(X, Y)
    
   #get the R2 
    MLR_truths =regOLS.predict(X)
    
    d= {}
    
    for i,c in enumerate(predictand_channels):
        Ycol = Y[:,i]
        MLRcol = MLR_truths[:,i]
        R2 = metrics.r2_score(Ycol, MLRcol)
        RMSE = metrics.mean_squared_error(Ycol, MLRcol, squared =False)
        MAE = metrics.mean_absolute_error(Ycol, MLRcol)
        d[c] = f'R2 = {R2:.4f}, RMSE = {RMSE:.4f}, MAE = {MAE:.4f}'
        log.info(f'{c} y variance explained by all predictors fit = ' + str(np.round(R2,5)))
        log.info(f'{c} RMSE = ' + str(np.round(RMSE,5)))
        log.info(f'{c} MAE = ' + str(np.round(MAE,5)))
        plotdata(Ycol, MLRcol, c,figdir, nick)
        
    p= predictand_channels[0] if len(predictand_channels) == 1 else 'ALL'
    with open (figdir / f'{nick}_MLR_{p}_eventlog.txt', 'w') as f:
        x=Path(filename).resolve().parent.name     
        print(datetime.now(), file = f)
        print(x, file = f)
        print(nick, file = f)
        print(d, file = f)
           

    #save the model
    picklename = figdir / f'{nick}_MLR_{p}.pickle'
    with open(picklename, 'wb') as f:
              pickle.dump(regOLS, f)

   



    

