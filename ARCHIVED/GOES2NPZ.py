#goes_to_pvisFINAL
import h5py
import numpy as np
import shutil
import matplotlib.pyplot as plt
import os
import re
from glob import glob
from netCDF4 import Dataset

#from dask.diagnostics import ProgressBar


def main():
    
    timesteps = find_timesteps()
    
    ALL_TIMES_BTS = []#this wil lbe the final array of all the time stamps and the BTs for each of the 4 channels
    
    #remaining functions needs to loop through this so we make for each time step
    for timestep in timesteps:
        filenames = get_files_from_timestep(timestep)
        if filenames is None:
            print('Warning: Data missing for timestep ' + timestep)
            continue
        
            #print ("these are the file names", filenames)
            #print("****************")
            #print("these are the SORTED file names", sorted(filenames))
   
        #function to make the BTs and save the data sets 
        
        c07_ABI_L1b = filenames[0]
        c11_ABI_L1b = filenames[1]
        c13_ABI_L1b = filenames[2]
        c15_ABI_L1b = filenames[3]
        print("C07 is" + c07_ABI_L1b + "\nC11 is" + c11_ABI_L1b + "\nC13 is" + c13_ABI_L1b + "\nC15 is" + c15_ABI_L1b)
        
        #calculates the BT and makes arrays of each then combines to 1 array X x Y x (4 channels) for the time stamp
        GOES_BT_array = goes_rad_to_BT(c07_ABI_L1b, c11_ABI_L1b, c13_ABI_L1b, c15_ABI_L1b )

        #C07arr = c07
        #C11arr = c11
        #C13arr = c13
        #C15arr = c15
       
        #make a single array of the 4 channels
       
        #GOES_BT_array = np.vstack((C07arr,C11arr, C13arr, C15arr))
        print(GOES_BT_array, np.nanmax(GOES_BT_array), np.nanmin(GOES_BT_array))
        print ('done with timestep'+ timestep + "continuing to next timestep")
        
   #save the npz set of 4 BTs for each time step
    ALL_TIMES_BTS = ALL_TIMES_BTS.append(GOES_BT_array)
    #ALL_TIMES_BTS['samples'] = [["timestep"] for timestep in timesteps]
    
    filename = f'GOESABICOLOCATION_TEST/GOES/GOES_BT_case.npz'
    np.savez_compressed(filename, **ALL_TIMES_BTS)
   

    
    
# this is designed to go through my file directory and find the unique times of data "timesteps" 
#that will be used to group the GOES channel data by    
        
######### NEED TO MANUALLY ADJUST FILE PATHS THROUGHOUT BEFORE RUNNING SINCE IVE LOST THE PATH ABILITY SOMEHOW
def find_timesteps():
    #paths = glob(os.path.expanduser('~/GOESABICOLOCATION_TEST/GOES/OR*'))
    paths = glob(os.path.expanduser('GOESABICOLOCATION_TEST/GOES/OR*'))
    print("my paths are", paths)
    timesteps = set()
    for path in paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        timestep = parts[3]
        timesteps.add(timestep)
    timesteps = sorted(timesteps)
    print ("these are the timesteps", timesteps, len(timesteps))
    return timesteps

# in theory this should then pull out the group of 4 files (ch 7/11/13/15) that occur at the same time
# to then run them through the goes to proxy code
def get_files_from_timestep(timestep):
    paths2 = glob(os.path.expanduser('GOESABICOLOCATION_TEST/GOES/OR*' + (timestep) + '*')) # if more than the 4 required files need to respecify glob for the 4 files i care about
    if len(paths2) != 4:
        return None
    else:
        #print("these are", paths2)
        #print("*****")
        #print("these are SORTED", sorted(paths2))
        SORTEDpaths = sorted(paths2)
        return SORTEDpaths

def goes_rad_to_BT(c07_filename, c11_filename, c13_filename, c15_filename):
        #save the BTs array result as a .nc with all the same attributes as a L1b file so can be read into ARCHER * SATPY
   # 1) Get brightness temperatures from each channel. end result is a variable with an array of all the BTs for that channel
    channels = ['C07', 'C11', 'C13', 'C15']
    #datas = 
    c07 = get_bt_from_goes_file(c07_filename)
    c11 = get_bt_from_goes_file(c11_filename)
    c13 = get_bt_from_goes_file(c13_filename)
    c15 = get_bt_from_goes_file(c15_filename)
    
    Cbands = c07 + c11 + c13 +c15
    Cband_arr = np.array(Cbands)
    return Cband_arr      
    

  
def get_bt_from_goes_file(goes_filename):
    goes_filename=os.path.expanduser(goes_filename)
    with h5py.File(goes_filename, "r") as in_file: 
        # Read radiance and the radiance meta-data out of the file.
        # The values aren't stored as actual radiance values. They're stored
        # in a compressed format to save disk space.  To get actual radiances, 
        # we'll have to multiply by a scale factor and then add
        # an offset.
        radiance_var = in_file['Rad']
        offset = radiance_var.attrs['add_offset']
        scale_factor = radiance_var.attrs['scale_factor']
        valid_min, valid_max = radiance_var.attrs['valid_range']
        # Copy the radiance values out of the file into our array.  We're using 
        # a 32 bit float array instead of the default 64 bit array to reduce 
        # memory usage and computation time.
        radiance = np.float32(radiance_var)

        # Read constants out of the file (again 32 bit)
        bc1 = np.float32(in_file['planck_bc1'])
        bc2 = np.float32(in_file['planck_bc2'])
        fk1 = np.float32(in_file['planck_fk1'])
        fk2 = np.float32(in_file['planck_fk2'])
    # We've read everything we need out of the file so we can go ahead and 
    # end our "with" block to close the file.

    # Get the indices where the data is valid.
    valid_indices = np.logical_and(radiance >= valid_min, radiance <=valid_max)

    # Finally, we'll apply the scale factor and offset to all the locations
    # where the data is valid.
    radiance[valid_indices] *= scale_factor
    radiance[valid_indices] += offset
    # Everywhere the data is not valid, we'll set to "not a number".
    radiance[~valid_indices] = np.nan
    
    # Convert to brightness temperatures.
    # We're going to use the following equation, but I've optimized it to 
    # reduce memory usage:
    # bt = ( fk2/(np.log((fk1/radiance) + 1))  -  bc1 )/bc2 
    bt = fk1/radiance
    bt += 1
    np.log(bt, out=bt)
    bt = fk2/bt
    bt -= bc1
    bt /= bc2   

    return bt
    
    
    
    
if __name__ == "__main__":
    main()


