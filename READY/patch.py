
import numpy as np
import itertools as it
from pathlib import Path
import sys
from common import log, rgb, reset, blue, orange, yellow, bold

def slices(lim, step):
    return [slice(a, a + step) for a in range(0, lim - step + 1, step)]
def all_patches(samples, x, y, patch_size):
    """Given the number of samples and the dimensions of each sample,
    compute all patch identifiers. Each patch is a 2-dimensional array.
    Using 256 as patch_size, these are 256x256 image patches. Each patch identifier
    looks like (3, slice(256, 512), slice(256, 512)). This example patch identifier
    describes going to the fourth sample (index 3), and slicing out the patch of the
    overall 2-d array from 256-512 for both rows and columns.
    A patch identifier is useful because it can be directly used as an index into
    the arrays. If an array of DNB data is of shape (20, 3000, 4000), meaning 20 samples
    each of which is a 3000x4000 2-d array, we can access a patch of data by:
    DNB[p] where p is a patch identifier."""
    return list(it.product(range(samples), slices(x, patch_size), slices(y, patch_size)))

def as_array(patches):
    """Given patch identifiers, convert each to a tuple of 5 numbers, so
    that they can be stored in a numpy array and saved for later."""
    return np.array([(sample, x.start, x.stop, y.start, y.stop) for sample, x, y in patches])

def mypatch(case, patchdims=512):
    shape = case['latitude'].shape
    channels = case['channels'] 
    varnum = len(channels)#number of variables
    num_samples = shape[0]
    rows = int(shape[1]/patchdims)
    columns = int(shape[2]/patchdims)
    print(f'I am in mypatch and this is the case info: original case size is {shape}, I have {num_samples} samples that are {rows} rows, {columns} columns, and size {patchdims},the number of variables is {varnum}'  )
    PATCHED = np.zeros(((num_samples*rows*columns),patchdims,patchdims,varnum))
    NEWARRAY = np.zeros((shape[0],shape[1],shape[2],varnum))
    
    for i in range (varnum):
        NEWARRAY[:,:,:,i] = case[channels[i]]
    patch_id = []
    whichpatch=0 #which patch we are at
    for i in range(num_samples):
        for ii in range(rows): 
            for jj in range(columns):
                i_start = ii*patchdims
                j_start = jj*patchdims
                PATCHED[whichpatch,:,:,:] = NEWARRAY[i,i_start:i_start+patchdims,j_start:j_start+patchdims,:]  
                ID = f'{case["samples"][i]}_{i_start}_{j_start}_{patchdims}'
                patch_id.append(ID)
                whichpatch=whichpatch+1
                #print('ID is', ID)
        print(f'i am done patching the {i+1} sample')
    array_data = {"PATCH_ID": patch_id}
    for i in range (varnum):
        array_data[channels[i]] = PATCHED[:,:,:,i]     
    return array_data
    
def patch_case(case, patch_size=512):
    shape = case['latitude'].shape
    a_patches = all_patches(shape[0], shape[1], shape[2], patch_size)
    arr_channels = case['channels']
    meta_channels = [ch for ch in case.files if ch not in arr_channels]
    log.info(f'Patching channels: {arr_channels}')
    arr_data = mypatch(case, patch_size)
    
    #arr_data = {c: np.stack([case[c][p] for p in a_patches], axis=-1)
    #           for c in arr_channels}

    metas = {c: case[c] for c in meta_channels}
    patch_samples = [case['samples'][p[0]] for p in a_patches]
    metas['samples'] = np.array(patch_samples)
    new_case = {**arr_data, **metas}
    return new_case


def patchcase(case):
    newcase = patch_case(case)
    return newcase
   
def patch(args):
    case = np.load(args.npz_path)
    patched = patch_case(case, args.PATCHSIZE)
    savepath = args.npz_path[:-4]+ "_PATCHED_" + str(args.PATCHSIZE) + ".npz"
    print("I am saving the case")
    np.savez_compressed(savepath,**patched )
    print(" I have saved the case as" + savepath)
    
