import numpy as np

def load_data( data_file ):

    print('Loading data from file: ' + data_file)
    my_file = np.load( data_file )

    print('Assigning to variables:')
    print('   Xdata_train, Ydata_train')
    Xdata_train = my_file['Xdata_train']
    Ydata_train = my_file['Ydata_train']
    print('   Xdata_test, Y_data_test')
    Xdata_test = my_file['Xdata_test']
    Ydata_test = my_file['Ydata_test']
    print('   Lat/Lon\n')
    Lat_train  = my_file['Lat_train']
    Lon_train  = my_file['Lon_train']
    Lat_test   = my_file['Lat_test']
    Lon_test   = my_file['Lon_test']

    if ('Xdata_scalar_train' in my_file) and ('Xdata_scalar_test' in my_file):
        Xdata_scalar_train = my_file['Xdata_scalar_train']
        Xdata_scalar_test = my_file['Xdata_scalar_test']
    else:
        Xdata_scalar_train = None
        Xdata_scalar_test = None

    return Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
        Lat_train, Lon_train, Lat_test, Lon_test, \
        Xdata_scalar_train, Xdata_scalar_test
