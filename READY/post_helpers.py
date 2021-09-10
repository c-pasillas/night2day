#import statements
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import gzip
import math
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

#channel stats
def basic_stats(dic, casename):
    for channel, x in dic.items():
        if x.mean() <1:
            
            print(f'{channel} min/med/max/mean/std {x.min():.2g}/{np.median(x):.2g}/{x.max():.2g}/{x.mean():.2g}/{x.std():.2g}')
            
        else:
            print(f'{channel} min/med/max/mean/std {x.min():.2f}/{np.median(x):.2f}/{x.max():.2f}/{x.mean():.2f}/{x.std():.2f}')
    Mchannels = [c for c in dic if c.startswith("M")]
    DNBchannels = [c for c in dic if not c.startswith("M")]
    Mcols = [dic[c].flatten() for c in Mchannels]
    DNBcols= [dic[c].flatten() for c in DNBchannels]
    logDNBcols = [np.log10(A) for A in DNBcols]
    #boxplot
    # figure related code
    fig = plt.figure()
    fig.suptitle(f' Mband Data Distributions for Case {casename}', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot(Mcols,labels=Mchannels, sym='' )
    ax.set_title('axes title')
    ax.set_xlabel('Channels') #xlabel
    ax.set_ylabel('Brightness Temperatures') #ylabel
#   plt.show()
    plt.savefig(f"{casename}_Mband_boxplot.png", bbox_inches='tight', dpi=500)#adjust better path later and name

    # figure related code
    fig = plt.figure()
    fig.suptitle(f' Radiance Data Distributions for Case {casename}', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot(DNBcols,labels=DNBchannels, sym='' )
    ax.set_title('axes title')
    ax.set_xlabel('Channels') #xlabel
    ax.set_ylabel('Radiances') #ylabel
#   plt.show()
    plt.savefig(f"{casename}_DNB_boxplot.png", bbox_inches='tight', dpi=500)#adjust better path later and name
    
    # figure related code
    fig = plt.figure()
    fig.suptitle(f' Log Radiance Data Distributions for Case {casename}', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot(logDNBcols,labels=DNBchannels, sym='' )
    ax.set_title('axes title')
    ax.set_xlabel('Channels') #xlabel
    ax.set_ylabel('Radiances') #ylabel
#   plt.show()
    plt.savefig(f"{casename}_logDNB_boxplot.png", bbox_inches='tight', dpi=500)#adjust better path later and na
    
    #violin plots
    #plt.violinplot(cols, showextrema=False);
    
    
def xy_relations(array1, array2):
    y_true = array1.flatten() #DNB
    y_pred = array2.flatten() #ML
     
    CC = np.corrcoef(y_true, y_pred)
    print( "the Pearson's Coorelation for DNB and MLdata is", CC)
    MSE =  mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)
    print( "the MSE for DNB and MLdata is", MSE)
    MAE = mean_absolute_error(y_true, y_pred)
    print( "the MAE for DNB and MLdata is", MAE)
    Explained_var = explained_variance_score(y_true, y_pred)
    print( "the Explained Variance for DNB and MLdata is", Explained_var)
    R2 = r2_score(y_true, y_pred)
    print( "the R2 for DNB and MLdata is", R2)
    
    data1 = y_true
    data2 = y_pred

    #calcualte the coorelaiton coffecticent
    from numpy import cov

    # calculate covariance matrix
    covariance = cov(data1, data2)
    print("covariance for DNB and MLdata is ", covariance)
    # calculate Pearson's correlation
    from scipy.stats import pearsonr
    corrp, _ = pearsonr(data1, data2)
    print('Pearsons correlation for DNB and MLdata: %.3f' % corrp)

    from scipy.stats import spearmanr
    # calculate spearman's correlation
    corrs, _ = spearmanr(data1, data2)
    print('Spearmans correlation for DNB and MLdata: %.3f' % corrs)

    from numpy import corrcoef
    COVAR = corrcoef(data1, data2)
    print("im printing COVAR shape")
    print(COVAR.shape)
    print("COVAR values for DNB and MLdata are", COVAR)
  
    
    
    
    
def cloudfree(truths):
    CLOUD = truths['DNB_log_FMN'].flatten()>0.5
    percent = np.count_nonzero(CLOUD)/len(CLOUD)#percent above the threshold given
    print(f"percent of the data that is \"cloud free\" using 0.5  {percent}")
          
          
#making images          

#helper functions for  COLE drawing
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

# #drawmy COLE data

def mkIMG ():
    import pathlib
    p = pathlib.Path("IMAGES")
    p.mkdir(exist_ok = True)
    return p
    
def draw_COLE(data_dic, name): 
    imagedir = mkIMG()
    for i in range(0,10): #len(truth['latitude'])): #patches
        print(f"starting patch {i}")
        for label in data_dic:
            image_array = data_dic[label][i]
            p = imagedir / f'{name}_{label}_{i}.png'
            if label == "DNBdiff":
                # plt.colorbar()
                plt.imsave(str(p), image_array, cmap = "seismic", vmin = -0.4, vmax = 0.4)  
            else:
                show_byte_img(scale(image_array), name = str(p))

###### making ERF images 

def ERF(array1, array2):

       ###order may need to be modified (ie when do I do thngs that requrie flatten vs thingsthat need array and how to reshpae to original shape (ie if i need flatten to calcualte the ERF values butthen reshape to plot)
    #ERF applications on ML and truth values for image translation
    log.info(f'starting the ERF processes')
    #load raw radiance values (ML and original) or from above unnormPRED unnormTRU
    
    rad = array1.flatten() #DNB #y_true
    ML_rad = array2.flatten() #ML #y_pred

    # take the radiances and use the ERF display image scale
    Rmax= 1.26e-10 
    Rmin=2e-11

    #ERF stats
    ERFimage_truth=  255 * np.sqrt(np.abs((rad[:]-Rmin)/(Rmax - Rmin)))
    x= ERFimage_truth
    print(f'ERF DNB min/med/max/mean/std {x.min():.2f}/{np.median(x):.2f}/{x.max():.2f}/{x.mean():.2f}/{x.std():.2f}')

    ERFimage_ML=  255 * np.sqrt(np.abs((ML_rad[:]-Rmin)/(Rmax - Rmin)))
    print(f'ERF ML_DNB min/med/max/mean/std {x.min():.2f}/{np.median(x):.2f}/{x.max():.2f}/{x.mean():.2f}/{x.std():.2f}')
    
    
    x=ERFimage_truth#[:2000]
    y=ERFimage_ML#[:2000]
    print(x.shape, y.shape)
    
    #do i need to reshape? 
    #calculate basic relaitons for ERF values
    xy_relations(x.flatten(),y.flatten())
    
    # plotting one image

    #img = x[0]
    #imgplot = plt.imshow(img, cmap='gray')
    #plt.title('ERF imagery truth')
    #plt.colorbar()
    #plt.show()

    #img2 = y[0]
    #imgplot= plt.imshow(img2, cmap='gray')
    #plt.title("ERF imagery ML")
    #plt.colorbar()
    #plt.show()
    #%%

    #plot all images?
    #def plotit():
    # for i in range(len(x)):
    #     x=ERFimage_truth[i]
    #     y=ERFimage_ML[i]
    #     a=np.linspace(0,4000,100)
    #     b=a
    #     #plt.plot(x,y)
    #     #plt.show()
    #     #print ('This is the end of Truth plot using paper full moon max/min')

    #     plt.figure()
    #     plt.plot(x,y,'o', markersize =1, color='black')
    #     plt.xlabel('Truth')
    #     plt.ylabel('ML_predicted')
    #     plt.plot(a,b, 'r--', label='perfect')
    #     plt.legend()
    #     plt.title('DNB Truth vs ML DNB ERF scaled BVIs' )
    #     #plt.show()
    #     log.info(f'saving comparison plot {i}')
    #     plt.savefig(f"comparison_at_{i}.png") 
    #     plt.clf()
    #     plt.close()
    #     log.info(f'done saving comparison plot {i}')

    #     img = x
    #     imgplot = plt.imshow(img, cmap='gray', vmin =1000, vmax=5000)

    #     plt.title('ERF imagery truth')
    #     plt.colorbar()
    #     #plt.show()
    #     plt.savefig(f"ERF_truth_at_{i}.png") 
    #     plt.clf()
    #     plt.close()



    #     img2 = y
    #     imgplot= plt.imshow(img2, cmap='gray', vmin=1500, vmax=5000)
    #     plt.title("ERF imagery ML")
    #     plt.colorbar()
    #     #plt.show()
    #     plt.savefig(f"1550ERF_ML_at_{i}.png") 
    #     plt.clf()
    #     plt.close()

                

# ###hexbins
# #hexbin instead of scatter plot

# #plt.hexbin(x, y, C=None, gridsize=100, bins=None, xscale='linear', yscale='linear', extent=None, cmap=None, norm=None, 
# #vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors='face', reduce_C_function=<function mean>, mincnt=None, marginals=False, *, data=None, **kwargs)

#     #plt.hexbin(x,y, edgecolors = None, label='normalized DNB')#alpha =.002,
#     #plt.xlabel('X value = Truth DNB')
#     #plt.ylabel('Y value = ML DNB')
#     #plt.legend()
#     #plt.title('truth DNB vs ML DNB for Full Moon Norm Radiances using M15 band')
#     #plt.colorbar()
#     #plt.show()


# #density scatter plot for larger datsets
# # from matplotlib import cm
# # from matplotlib.colors import Normalize 
# # from scipy.interpolate import interpn

# def density_scatter( x , y, ax = None, sort = True, bins = 1000, **kwargs )   :
#     """
#     Scatter plot colored by 2d histogram
#     """
#     if ax is None :
#         fig , ax = plt.subplots()
#     data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
#     z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

#     #To be sure to plot all data
#     z[np.where(np.isnan(z))] = 0.0

#     # Sort the points by density, so that the densest points are plotted last
#     if sort :
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]

#     ax.scatter( x, y, c=z, **kwargs )

#     norm = Normalize(vmin = np.min(z), vmax = np.max(z))
#     cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
#     cbar.ax.set_ylabel('Density')
#     #add labels
#     return ax

# density_scatter( x, y, bins = [30,30] )

