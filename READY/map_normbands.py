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


def plot_normband(args):
    print("im in make norm band plots and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    log.info(f'I am making the plots')

    M13norm = case['M13norm']
    M14norm = case['M14norm']
    M15norm = case['M15norm']
    M16norm = case['M16norm']
    Reflectance = case['SM_reflectance']
    samples = case['samples']
    #print(samples)
    
    for i in range(len(samples)):
    #for i in range(3):   
        iM13 = M13norm[i,:,:]
        iM14 = M14norm[i,:,:]
        iM15 = M15norm[i,:,:]
        iM16 = M16norm[i,:,:]
        iRef = Reflectance[i,:,:]
        sampletime = samples[i]
        Iref_maxval =np.nanmax(iRef)
        Iref_minval = np.nanmin(iRef)
        
        # just do each channel versus multi loops
        img = iM13
        imgplot = plt.imshow(img, cmap='gray_r')#, vmin =0, vmax=1)

        plt.title('Raw M13norm')
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M13norm_{i}_{sampletime}.png") 
        plt.clf()
        plt.close()

        img2 = iM14
        imgplot= plt.imshow(img2, cmap='gray_r')#, vmin=0, vmax=1)
        plt.title("Raw M14norm")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M14_norm_{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        img3 = iM15
        imgplot= plt.imshow(img3, cmap='gray_r')#, vmin=0, vmax=1)
        plt.title("Raw M15norm")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M15norm__{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        img4 = iM16
        imgplot= plt.imshow(img4, cmap='gray_r')#, vmin=0, vmax=1)
        plt.title("Raw M16")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M16norm__{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        img5 = iRef
        imgplot= plt.imshow(img5, cmap='gray')#, vmin=0, vmax=1)
        plt.title("SM_Reflectance")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_SMreflectance__all_{i}_{sampletime}.png") 
        plt.clf()
        plt.close()

        img6 = iRef
        imgplot= plt.imshow(img6, cmap='gray', vmin=0, vmax=100)
        plt.title("SM_Reflectance clipped")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_SMreflectance__clip_{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        img7 = iRef
        imgplot= plt.imshow(img7, cmap='gray', vmin=Iref_minval, vmax=Iref_maxval)
        plt.title("SM_Reflectance scaled")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_SMreflectance__scaled_{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
