from matplotlib import pyplot as plt
import numpy as np

def setup_performance_diagram():

#    nbins = 100
#    dbin = 0.01
    nbins = 1000
    dbin = 0.001

    csi = np.zeros((nbins,nbins))
    hrs = np.zeros((nbins,))
    fals = np.zeros((nbins,))
    bias = np.zeros((nbins,nbins))
    for ihit in range(nbins):
        for ifal in range(nbins):
            hr = ihit*dbin
            far = ifal*dbin
            try:
                csi[ihit,ifal] = 1.0/( (1.0/hr) + (1.0/(1.0-far)) -1.0 )
                bias[ihit,ifal] = hr/(1.0-far)
            except ZeroDivisionError:
                csi[ihit,ifal] = np.nan
                bias[ihit,ifal] = np.nan
            hrs[ihit] = hr
            fals[ifal] = far

    csilevels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    csicolor = 'black'
    csistyle = 'dashed'
    cs = plt.contour(1.0-fals,hrs,csi,colors=csicolor,linestyles=csistyle,levels=csilevels)
    plt.clabel(cs,csilevels,fontsize='x-small',colors=csicolor,fmt='%3.1f')

    biaslevels = [1./10., 1./5., 1./3., 1./2., 1./1.5, 1./1.25, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]

    biascolor = 'grey'
    biasstyle = 'dotted'
    cs = plt.contour(1.0-fals,hrs,bias,colors=biascolor,linestyles=biasstyle,levels=biaslevels)
    cs = plt.contour(1.0-fals,hrs,bias,colors=biascolor,linestyles='solid',levels=[1.0])
    for alev in biaslevels:
        if alev <= 1.0:
            plt.text(alev,1.0,'{0:4.2f}'.format(1.0/alev),va='center',ha='center',color=biascolor)
        else:
            plt.text(1.0,1.0/alev,'{0:4.2f}'.format(1.0/alev),va='center',ha='center',color=biascolor)

    plt.xlabel('Success Ratio (1-FAR)')
    plt.ylabel('Probability of Detection')
    
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid()

    return
