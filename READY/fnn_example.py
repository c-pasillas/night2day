
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import math
import tensorflow.keras.layers as layers

##CNN (For later)
#from keras.models import Model
#from keras.optimizers import RMSprop
#from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv3D,MaxPooling2D,UpSampling3D,Conv2DTranspose,Activation, Lambda,MaxPooling3D,Conv2D
#from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional_recurrent import ConvLSTM2D
#from keras.layers import TimeDistributed, LayerNormalization
#from keras.models import Model,Sequential
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adadelta, RMSprop,SGD,Adam
#from keras import regularizers
#from keras.regularizers import l2
#from keras import backend as K
#from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers import SpatialDropout2D,Reshape
#from keras.layers.merge import concatenate
#from keras.optimizers import SGD
#from keras.layers import Bidirectional
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

n_units = 64
model = tf.keras.Sequential()
# First hidden layer:
model.add(layers.Dense(n_units, activation='relu', input_shape=(1,) ))
# Second hidden layer:
model.add(layers.Dense(n_units, activation='relu'))
# Output layer:  just 1 node and no activation function
model.add(layers.Dense(1))
model.summary()

#model.compile(optimizer=keras.optimizers.Adam(0.01),  # Adam optimizer
#            loss='mse',       # mean squared error
#            metrics=['mae'])  # mean absolute error

#n_epochs = 200   # 250
#history = model.fit(x, y, validation_split=0.30, epochs=n_epochs, batch_size=128)




