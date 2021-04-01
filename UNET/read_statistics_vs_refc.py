import numpy as np
import sys

panels = ['F','G','H','I','J']
model_name = {}
model_name['F'] = 'C13'
model_name['G'] = 'C13+GLM'
model_name['H'] = 'C13+C09'
model_name['I'] = 'C13+C09+GLM'
model_name['J'] = 'C13+C09+C07+GLM'
nmodels = len(panels)
indices = [2+i for i in range(nmodels)]

def read_statistics_vs_refc(filename='statistics_vs_refc.txt'):

    stats = {}

    f = open(filename,'r')

    aline = f.readline().strip()
    items = aline.split('=')
    if items[0] != 'nrefs': sys.exit('error1')
    nref = int(items[1])

    stats['ref'] = np.zeros((nref,))
    stats['numobs'] = np.zeros((nref,))

    nstat = 2
    stats['rmsd'] = {}
    stats['rsq'] = {}
    for apanel in panels:
        stats['rmsd'][model_name[apanel]] = np.zeros((nref,))
        stats['rsq'][model_name[apanel]] = np.zeros((nref,))

    astat = f.readline().strip()
    if astat != 'numobs': sys.exit('error2')
    for iref in range(nref):
        items = [anitem.strip() for anitem in f.readline().strip().split()]
        if len(items) != 3: sys.exit('error3')
        i = int(items[0])
        r = float(items[1])
        v = float(items[2])
        if i != iref: sys.exit('error4')
        stats['ref'][iref] = r
        stats[astat][iref] = v

    for istat in range(nstat):
        astat = f.readline().strip()
        for iref in range(nref):
            items = [anitem.strip() for anitem in f.readline().strip().split()]
            if len(items) != 2+nmodels: sys.exit('error5')
            i = int(items[0])
            r = float(items[1])
            if i != iref: sys.exit('error4')
            for apanel,indx in zip(panels,indices):
                stats[astat][model_name[apanel]][iref] = float(items[indx])

    f.close()

    return stats
