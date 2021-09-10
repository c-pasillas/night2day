import numpy as np
from pathlib import Path
import itertools as it
import sqlite3
import time
import common
from common import log, reset, blue, yellow, orange, bold 

all_predictands = ['DNB_full_moon_norm',
                   'DNB_log_full_moon_norm']

all_predictors = [   'BTD1213norm',
                         'BTD1214norm',
                         'BTD1215norm',
                         'BTD1216norm',
                         'BTD1314norm',
                         'BTD1315norm',
                         'BTD1316norm',
                         'BTD1415norm',
                         'BTD1416norm',
                         'BTD1516norm',
                         'M12norm',
                         'M13norm',
                         'M14norm',
                         'M15norm',
                         'M16norm',]

predictand = 'DNB_full_moon_norm' #, 'DNB_log_full_moon_norm'

predictors = [   'BTD1213norm',
                         'BTD1214norm',               
                         'BTD1415norm',                     
                         'BTD1516norm',
                         'M12norm',
                         'M13norm',
                         'M14norm',
                         'M15norm',
                         'M16norm',]

ts = 0.2
rs = 10

PREDICTAND = predictand
PREDICTORS = list(predictors)

def prep (case):
    TAND = case[predictand]
    TORS = np.stack([case[c] for c in predictors], axis = -1)
    newcase = {"TORS": TORS, 
               "TAND":TAND, 
              "Predictors": PREDICTORS,
              "Predictand": PREDICTAND}
    return newcase

def FINALprep(args):
    case = np.load(args.npz_path)
    print("I loaded the case")
    Final_case = prep(case)
    print("I am now saving case")
    savepath = args.npz_path[:9]+ "_MODELready.npz"
    np.savez(savepath,**Final_case)


