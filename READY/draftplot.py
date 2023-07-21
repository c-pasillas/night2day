
#  this will then go into post processing helper functions
#need something that makes just basic gray scale plots of the raw IR bands so we can visually compare
#input is the original bands array.npz  Ref_ALLcase
#import statements
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import gzip
import math
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from common import log




def plot_raw(args):
    print("im in make plots and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    log.info(f'I am making the plots')

    M13 = case['M13']
    M14 = case['M14']
    M15 = case['M15']
    M16 = case['M16']
    Reflectance = case['SM_reflectance']
    samples = case['samples']

    for i in range(len(samples)):
        iM13 = M13[i]
        iM14 = M14[i]
        iM15 = M15[i]
        iM16 = M16[i]
        iRef = Reflectance[i]
        
        # jsut do each channel versus multi loops
        img = iM13
        imgplot = plt.imshow(img, cmap='gray', vmin =0, vmax=1)

        plt.title('Raw M13')
        plt.colorbar()
        #plt.show()
        plt.savefig(f"Lunar_case_Raw_M13_{i}.png") 
        plt.clf()
        plt.close()

        img2 = iM14
        imgplot= plt.imshow(img2, cmap='gray', vmin=0, vmax=1)
        plt.title("Raw M14")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"Lunar_case_raw_M14__{i}.png") 
        plt.clf()
        plt.close()
        
        img3 = iM15
        imgplot= plt.imshow(img3, cmap='gray', vmin=0, vmax=1)
        plt.title("Raw M15")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"Lunar_case_raw_M15__{i}.png") 
        plt.clf()
        plt.close()
        
        img4 = iM16
        imgplot= plt.imshow(img4, cmap='gray', vmin=0, vmax=1)
        plt.title("Raw M15")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"Lunar_case_raw_M15__{i}.png") 
        plt.clf()
        plt.close()
        
        img5 = iRef
        imgplot= plt.imshow(img5, cmap='gray', vmin=0, vmax=1)
        plt.title("SM_Reflectance")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"Lunar_case_raw_SMreflectance__{i}.png") 
        plt.clf()
        plt.close()