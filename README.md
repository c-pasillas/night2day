# night2day
to do list 

config file to set paths call the config file in each processing step if necessary
soln: run from where rw data is

POSTPROCESSING
- need to have a code to rebuild the larger array ( from patches at each time step).
  This way I can have single image of the area for test cases.
  Later I would be running the learned model on different data set and can hopefully
  reprocess the image by time and area as well.

# Get started
The `main.py` file (probably found in the `READY` folder) can be run as a program
from the command line. For example, if the path to main is `/a/b/c/night2day/READY/main.py`
then we can run the program and ask for its help message like:
```shell
/a/b/c/night2day/READY/main.py --help
```

You can, of course, set up an alias for this, or put the program on your shell's path.
**If you alias it** as `night2day` (a good name for this program),
running the program and asking for the help message would look like:
```shell
night2day --help
```

In any case, you use the program on a folder of `.h5` files. You should have a
folder containing a directory of `.h5` files named `raw-data` or `RAWDATA` or
some other name staring with `raw`. If you are in such a root folder and listed the contents
you would expect to see something like this:
```
├── raw-data
│   ├── GDNBO-SVDNB_npp_d20190319_t1239033_e1244419_b38294_c20200513155230218100_nobc_ops.h5
│   ├── GDNBO-SVDNB_npp_d20190320_t1039074_e1044478_b38307_c20200513155302497143_nobc_ops.h5
│   ├── GDNBO-SVDNB_npp_d20190320_t1358170_e1403556_b38309_c20200513155349263310_nobc_ops.h5
│   ├── GMODO-SVM08-SVM09-SVM10-SVM11-SVM12-SVM13-SVM14-SVM15-SVM16_npp_d20190319_t1239033_e1244419_b38294_c20200513155233050127_nobc_ops.h5
│   ├── GMODO-SVM08-SVM09-SVM10-SVM11-SVM12-SVM13-SVM14-SVM15-SVM16_npp_d20190320_t1039074_e1044478_b38307_c20200513155226104073_nobc_ops.h5
│   └── GMODO-SVM08-SVM09-SVM10-SVM11-SVM12-SVM13-SVM14-SVM15-SVM16_npp_d20190320_t1358170_e1403556_b38309_c20200513155349916445_nobc_ops.h5
```
You can see here that the current directory contains a folder named `raw-data`, which holds various `.h5` files,
which are the raw sensor data collected from a satellite. The filenames encode various information,
like which sensor data it contains, and when exactly it was captured.

In that current directory (the parent of the `raw` folder), run the first step like:
```shell
night2day pack-case
```
This command will find all the `.h5` files in the `raw` folder, match them up by time,
colocated the various channels, save intermediate results like pictures and array data,
and ultimately write out the file `case.npz`, which is a `Numpy` array file containing all 
the available channels and samples from the `.h5` files.

Next, run:
```shell
night2day normalize
```
This command will use the original channels to compute several derived channels
that are normalized using constants appropriate to each sensor. It ultimately writes
out the file `case_norm.npz`, a `Numpy` array file with all the original channels,
plus all the newly calculated normalized channels.

Now we are ready to prepare the inputs to the machine learning model. Run:
```shell
night2day learn-prep
```
This will write out all the available channels to a file `learning_channels.txt`,
and ask you to edit this file. As the first lines of this file explain, you select 
which channel to predict by using an asterisk `*` next to its name, and you
ignore channels by commenting out their names with the pound `#`.

Once you edit this file to select the predictor channels and the predicted channel,
run this again:
```shell
night2day learn-prep
```
The program will read `learning_channels.txt` and `case_norm.npz`, segment the
sensor data into small chunks of 2-dimensional data called `patches`, throw out
patches that are outside the area-of-interest, and create shuffled collections
of patches for training and testing the machine learning model.

In the folder `ML_INPUT`, it will create a folder representing the current
channel selection, and write out a file `data.npz` containing the array data
for training a machine learning model, along with a description of the data.
This `data.npz` file is ready to be used with a machine learning process
developed previously in our lab.

For any of the `.npz` files produced by this program, you can access
and inspect them using `Numpy`. Interactively you can see what arrays a
`.npz` file contains like:
```python
import numpy as np
f = np.load('data.npz')

f.files =>
['Lat_train',
 'Lon_train',
 'Xdata_train',
 'Ydata_train',
 'Lat_test',
 'Lon_test',
 'Xdata_test',
 'Ydata_test',
 'samples',
 'train_patches',
 'test_patches',
 'y_channel',
 'x_channels']

f['Xdata_train'].shape =>
(373, 256, 256, 12)
```

In this example, we can see that the file `data.npz` produced by the learn-prep step
contains many `Numpy` arrays for us to use. The array named `Xdata_train` contains
373 patches, each 256 by 256, each point being 12 channels deep.

Of course, using it in a program, you will want the file to be closed automatically,
so you should use the `with` construct like:
```python
with np.load('data.npz') as f:
    print(f.files)
    print(f['Xdata_train'].shape)
    # more computations with the arrays
```

Or, if you want to just load all arrays into memory at once, you can do this:
```python
with np.load('data.npz') as f:
    array_data = dict(f)

print(array_data.keys())
print(array_data['Xdata_train'].shape)
# more computations with the arrays
```

