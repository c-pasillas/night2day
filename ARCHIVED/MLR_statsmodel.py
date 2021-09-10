#!/usr/bin/env python
# coding: utf-8

# In[8]:
##### DO NOT USE, SM wont work for the size arrays i have
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import sys
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import style
from common import log

style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (20,10)

#determine what we care about
predict_channels = ['M12norm','M13norm','M14norm','M15norm','M16norm','BTD1215norm','BTD1216norm','BTD1315norm',
'BTD1316norm','BTD1415norm','BTD1416norm','BTD1516norm']

predictand_channels = ['DNB_norm', 'DNB_full_moon_norm', 'DNB_new_moon_norm', 'DNB_log_norm','DNB_log_full_moon_norm', 'DNB_log_new_moon_norm',
       'DNB_log_Miller_full_moon']

def MLR(npzfile, DNB_channel):
    log.info("starting MLR job")
    #load case
    f= np.load(npzfile)
    #chosen_predictor = predictand_channels[0]
    chosen_predictor = DNB_channel
    all_channels = predict_channels + [chosen_predictor]
    
    #pull the data we care about into pandads dFs
    # define the data/predictors as the pre-set feature names  
    predictors = pd.DataFrame({key: f[key].flatten() for key in predict_channels})

    # Put the target  in another DataFrame
    target = pd.DataFrame({chosen_predictor: f[chosen_predictor].flatten()}) 

    all_data = pd.DataFrame({key: f[key].flatten() for key in all_channels})
    log.info('data loaded')
  
    #plot the pairs
    #sb.pairplot(predictors)
    #plt.savefig(f'pairplots_for_{chosen_predictor}.png')
    #log.info('done making pairplots')
    
    #training the models
    log.info('training models')
    X1 = sm.add_constant(predictors)
    X = predictors
    y = target
    #MLR with intercept made
    model2 = sm.OLS(y, X1).fit()
    model2.save("MLR_intercept.pickle")

    #now try MLR at origin
    model = sm.OLS(y, X).fit()
    model.save("MLR_origin.pickle")

    log.info('saving model summaries')
    
    summaryimage(model, "origin")
    summaryimage(model2, "intercept")

def summaryimage (model, label):
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'MLR_{label}_summary.png')
    plt.close()
   
