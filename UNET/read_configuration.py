from ast import literal_eval

def read_configuration(filename='configuration.txt'):

    thekey = None

    config = {}

    print('reading configuration file=',filename)

    f = open(filename,'r')

    for aline in f:

        sline = aline.strip()
        if sline.startswith('#'): continue  #skip comments
        if len(sline) == 0: continue  #skip blank lines

        #format of a line is: variable=value
        items = [anitem.strip() for anitem in sline.split('=')]
        if len(items) != 2: continue  #bad format
        akey = items[0]
        avalue = items[1]

        try:
            avalue = literal_eval(avalue)
        except ValueError:
            pass

        if akey == 'my_file_prefix':
            config[avalue] = {}
            thekey = avalue
            continue

        if thekey == None:
            config[akey] = avalue
        else:
            config[thekey][akey] = avalue

    f.close()

    return config
