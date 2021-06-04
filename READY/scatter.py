#!/usr/bin/env python
# coding: utf-8


#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
from common import log
from pathlib import Path


mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
dpiFig = 300.

style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (20,10)

#%%   
def downsample (arr,samplesize):
    arr = arr.reshape(-1,2)
    rng = np.random.default_rng()
    #rng.shuffle(arr)
    a = arr[:int(samplesize)]
    return a[...,0].flatten(), a[...,1].flatten()      
    
def scatter (npz_filename, nick, samplesize) :
    log.info(f"starting scatterplotting postprocessing")
    outputdir= Path(npz_filename).resolve().parent
    f = np.load(npz_filename)
    shape = f['DNB_norm'].shape
        
    for c in f.files:
        log.info(f'starting scatter plot for {c}')
        truth, MLR = downsample(f[c], samplesize)        
        plt.scatter(truth, MLR, s=1, marker='.', alpha=0.5)
        plt.xlabel('DNB truth')
        plt.title(f"Radiance comparison for {c}")
        plt.ylabel('MLR truth') 
        plt.axline((truth[0],truth[0]), slope = 1, linewidth=1)
        #plt.ylim([-1,1])
        #plt.xlim([-1,1])
        plt.savefig(outputdir / f"{nick}_{c}_scatterplot_slope{samplesize}.png")
        plt.close()
        