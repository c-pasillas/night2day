from datetime import datetime
import sys

def read_datetimes(filename):

    datetimes = {'TRAIN':[], 'TEST':[]}

    f = open(filename,'r')

    for aline in f:

        if aline.startswith('*'):
            akey = aline.replace('*','').strip()
            continue
        
        items = [anitem.strip() for anitem in aline.split()]
        if len(items) != 2: sys.exit('error in read_datetimes')
        
        samplenumber = int(items[0])
        adatetime = datetime.strptime(items[1], '%Y%m%d%H%MZ')

        datetimes[akey].append(adatetime)

    f.close()

    return datetimes
