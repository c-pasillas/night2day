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
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from common import log


def plot_ML(args):
    print("im in make plots and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    log.info(f'I am making the plots')
    ML = case['MLtruth']
    MLcount =np.isnan(ML)
    yesnanML = np.count_nonzero(MLcount)
    totalML = len(ML)*3056*4064
    percentML = yesnanML/totalML
    print(yesnanML, totalML, percentML)
    for i in range(len(ML)):
    #for i in range(3):
        iML = ML[i,:,:]    
        maxvalML =np.nanmax(iML)
        minvalML = np.nanmin(iML)
        print(maxvalML, minvalML)
        
        MLimg = iML
        imgplot = plt.imshow(MLimg, cmap='gray', vmin =0, vmax=100)
        plt.title('ML-NVI Reflectance')
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)_ML_lunar_reflectance__{i}.png") 
        plt.clf()
        plt.close()