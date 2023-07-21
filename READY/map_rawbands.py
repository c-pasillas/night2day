
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


def plot_raw(args):
    print("im in make plots and my args are", args)
    case = np.load(args.npz_path)
    log.info(f'I loaded the case')
    log.info(f'I am making the plots')

    M13 = case['M13']
    M14 = case['M14']
    M15 = case['M15']
    M16 = case['M16']
    #Reflectance = case['SM_reflectance']
    samples = case['samples']
    #print(samples)
    
    for i in range(len(samples)):
    #for i in range(3):   
        iM13 = M13[i,:,:]
        iM14 = M14[i,:,:]
        iM15 = M15[i,:,:]
        iM16 = M16[i,:,:]
        #iRef = Reflectance[i,:,:]
        sampletime = samples[i]
        #Iref_maxval =np.nanmax(iRef)
        #Iref_minval = np.nanmin(iRef)
        
        # just do each channel versus multi loops
        img = iM13
        imgplot = plt.imshow(img, cmap='gray_r')#, vmin =0, vmax=1)

        plt.title('Raw M13')
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M13_{i}_{sampletime}.png") 
        plt.clf()
        plt.close()

        img2 = iM14
        imgplot= plt.imshow(img2, cmap='gray_r')#, vmin=0, vmax=1)
        plt.title("Raw M14")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M14__{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        img3 = iM15
        imgplot= plt.imshow(img3, cmap='gray_r')#, vmin=0, vmax=1)
        plt.title("Raw M15")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M15__{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        img4 = iM16
        imgplot= plt.imshow(img4, cmap='gray_r')#, vmin=0, vmax=1)
        plt.title("Raw M16")
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{args.npz_path(:-4)}_Raw_M16__{i}_{sampletime}.png") 
        plt.clf()
        plt.close()
        
        #img5 = iRef
        #imgplot= plt.imshow(img5, cmap='gray')#, vmin=0, vmax=1)
        #plt.title("SM_Reflectance")
        #plt.colorbar()
        #plt.show()
        #plt.savefig(f"{args.npz_path(:-4)}_Raw_SMreflectance__all_{i}_{sampletime}.png") 
        #plt.clf()
        #plt.close()

        #img6 = iRef
        #imgplot= plt.imshow(img6, cmap='gray', vmin=0, vmax=100)
        #plt.title("SM_Reflectance clipped")
        #plt.colorbar()
        #plt.show()
        #plt.savefig(f"{args.npz_path(:-4)}_Raw_SMreflectance__clip_{i}_{sampletime}.png") 
        #plt.clf()
        #plt.close()
        
        #img7 = iRef
        #imgplot= plt.imshow(img7, cmap='gray', vmin=Iref_minval, vmax=Iref_maxval)
        #plt.title("SM_Reflectance scaled")
        #plt.colorbar()
        #plt.show()
        #plt.savefig(f"{args.npz_path(:-4)}_Raw_SMreflectance__scaled_{i}_{sampletime}.png") 
        #plt.clf()
        #plt.close()


#def plot_raw(args):
 #   print("im in make plots and my args are", args)
  #  case = np.load(args.npz_path)
   # log.info(f'I loaded the case')
    #log.info(f'I am making the plots')
    #original = case['samples']
    #for i in range(len(original)):
     #   IR13=case['M13'][i]
      #  IR14=case['M14'][i]
       # IR15=case['M15'][i]
        #IR16=case['M16'][i]
        #ideally nest it but not sure since been so long since i coded
        #IRlist = ['IR13', 'IR14', 'IR15', 'IR16']
            #ideally nest it but not sure since been so long since i coded
        #for j in IRlist:
         #   img = [j]
          #  imgplot = plt.imshow(img, cmap='gray', vmin =0, vmax=1)
           # plt.title('{j} truth')
            #plt.colorbar()
                #plt.show()
            #plt.savefig(f"{j}_truth_at_{i}.png") 
            #plt.clf()
            #plt.close()