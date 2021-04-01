from matplotlib import pyplot as plt
import numpy as np

from read_statistics import read_statistics

statistics = read_statistics()

#model_name['I'] = 'C13+C09+GLM'
#model_name['J'] = 'C13+C09+C07+GLM'

fig = plt.figure()

fmt = {}
fmt['rmsd'] = '{0:6.3f}'
fmt['rsq'] = '{0:6.4f}'
fmt['csi20'] = '{0:5.3f}'
fmt['csi35'] = '{0:5.3f}'
fmt['pod20'] = '{0:5.3f}'
fmt['pod35'] = '{0:5.3f}'
fmt['far20'] = '{0:5.3f}'
fmt['far35'] = '{0:5.3f}'

def make_plot(xvar,yvar,stat):
    plt.scatter( statistics[xvar][stat], statistics[yvar][stat] )
    plt.grid()
    xmean = np.nanmean(statistics[xvar][stat])
    ymean = np.nanmean(statistics[yvar][stat])
    plt.scatter([xmean],[ymean],c='red')
    if stat == 'rmsd':
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.plot([0,10],[0,10],linestyle='dashed',color='black')
        plt.xlabel(xvar+' RMSD (dBZ)')
        plt.ylabel(yvar+' RMSD (dBZ)')
        #plt.text(9.5,0.5,xvar+' mean = '+fmt[stat].format(np.nanmean(statistics[xvar][stat])), ha='right')
        #plt.text(0.5,9.5,yvar+'  mean = '+fmt[stat].format(np.nanmean(statistics[yvar][stat])), ha='left')
    elif stat in ['rsq','csi20','csi35','pod20','pod35','far20','far35']:
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot([0,1],[0,1],linestyle='dashed',color='black')
        plt.xlabel(xvar+' '+stat.upper())
        plt.ylabel(yvar+' '+stat.upper())
        #plt.text(0.95,0.05,xvar+' mean = '+fmt[stat].format(np.nanmean(statistics[xvar][stat])), ha='right')
        #plt.text(0.05,0.95,yvar+'  mean = '+fmt[stat].format(np.nanmean(statistics[yvar][stat])), ha='left')
    figname = '../output/statistics_figures/fig_'+stat+'_'+yvar+'_vs_'+xvar+'.png'
    fig.savefig(figname)
    plt.clf()

make_plot('C13',            'C13+GLM','rmsd')
make_plot('C13',            'C13+C09','rmsd')
make_plot('C13+GLM',        'C13+C09','rmsd')
make_plot('C13+C09+C07+GLM','C13+C09','rmsd')

make_plot('C13',            'C13+GLM','rsq')
make_plot('C13',            'C13+C09','rsq')
make_plot('C13+GLM',        'C13+C09','rsq')
make_plot('C13+C09+C07+GLM','C13+C09','rsq')

make_plot('C13',            'C13+GLM','csi20')
make_plot('C13',            'C13+C09','csi20')
make_plot('C13+GLM',        'C13+C09','csi20')
make_plot('C13+C09+C07+GLM','C13+C09','csi20')

make_plot('C13',            'C13+GLM','csi35')
make_plot('C13',            'C13+C09','csi35')
make_plot('C13+GLM',        'C13+C09','csi35')
make_plot('C13+C09+C07+GLM','C13+C09','csi35')

make_plot('C13',            'C13+GLM','pod20')
make_plot('C13',            'C13+C09','pod20')
make_plot('C13+GLM',        'C13+C09','pod20')
make_plot('C13+C09+C07+GLM','C13+C09','pod20')

make_plot('C13',            'C13+GLM','pod35')
make_plot('C13',            'C13+C09','pod35')
make_plot('C13+GLM',        'C13+C09','pod35')
make_plot('C13+C09+C07+GLM','C13+C09','pod35')

make_plot('C13',            'C13+GLM','far20')
make_plot('C13',            'C13+C09','far20')
make_plot('C13+GLM',        'C13+C09','far20')
make_plot('C13+C09+C07+GLM','C13+C09','far20')

make_plot('C13',            'C13+GLM','far35')
make_plot('C13',            'C13+C09','far35')
make_plot('C13+GLM',        'C13+C09','far35')
make_plot('C13+C09+C07+GLM','C13+C09','far35')

