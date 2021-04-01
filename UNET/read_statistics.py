from datetime import datetime
import sys

panels = ['F','G','H','I','J']
model_name = {}
model_name['F'] = 'C13'
model_name['G'] = 'C13+GLM'
model_name['H'] = 'C13+C09'
model_name['I'] = 'C13+C09+GLM'
model_name['J'] = 'C13+C09+C07+GLM'
indices = [2,3,4,5,6]

def read_statistics(filename='statistics.txt'):

    statistics = {}
    statistics['samplenumber'] = []
    statistics['datetime'] = []
    for apanel in panels:
        statistics[model_name[apanel]] = {}
        statistics[model_name[apanel]]['rmsd'] = []
        statistics[model_name[apanel]]['rsq'] = []
        statistics[model_name[apanel]]['csi20'] = []
        statistics[model_name[apanel]]['csi35'] = []
        statistics[model_name[apanel]]['pod20'] = []
        statistics[model_name[apanel]]['pod35'] = []
        statistics[model_name[apanel]]['far20'] = []
        statistics[model_name[apanel]]['far35'] = []

    f = open(filename,'r')

    for aline in f:

        items = [anitem.strip() for anitem in aline.split(' ')]
        if len(items) != 7: 
            print(items)
            sys.exit('error1 in read_statistics')

        samplenumber = int(items[0])
        adatetime = datetime.strptime(items[1],'%Y%m%d%H%MZ')

        statistics['samplenumber'].append(samplenumber)
        statistics['datetime'].append(adatetime)

        for apanel,indx in zip(panels,indices):
            bitems = [anitem.strip() for anitem in items[indx].split(',')]
            if len(bitems) != 9: sys.exit('error2 in read_statistics')
            rmsd = float(bitems[0])
            rsq = float(bitems[1])
            rmax = float(bitems[2])
            csi20 = float(bitems[3])
            csi35 = float(bitems[4])
            pod20 = float(bitems[5])
            pod35 = float(bitems[6])
            far20 = float(bitems[7])
            far35 = float(bitems[8])
            statistics[model_name[apanel]]['rmsd'].append( rmsd )
            statistics[model_name[apanel]]['rsq'].append( rsq )
            statistics[model_name[apanel]]['csi20'].append( csi20 )
            statistics[model_name[apanel]]['csi35'].append( csi35 )
            statistics[model_name[apanel]]['pod20'].append( pod20 )
            statistics[model_name[apanel]]['pod35'].append( pod35 )
            statistics[model_name[apanel]]['far20'].append( far20 )
            statistics[model_name[apanel]]['far35'].append( far35 )

    f.close()

    return statistics
