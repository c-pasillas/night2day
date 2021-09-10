# This file contains all custom elements for the NN model.
#
# Note that when the model is saved, these functions are NOT saved,
# so they must be loaded explicitly when restoring the model from file!

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    import tensorflow.keras.backend as K  # needed for custom loss function
    import tensorflow as tf

################################################################
# Custom metric

def my_r_square_metric(y_true,y_pred):
   ss_res = K.sum(K.square(y_true-y_pred))
   ss_tot = K.sum(K.square(y_true-K.mean(y_true)))
   return ( 1 - ss_res/(ss_tot + K.epsilon()) )
   
################################################################
# Custom loss

def my_mean_squared_error_noweight(y_true,y_pred):
    return K.mean( tf.square(tf.subtract(y_pred,y_true)) )

def my_mean_squared_error_weighted1(y_true,y_pred):
    return K.mean( tf.multiply( tf.exp(tf.multiply(5.0,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )

def my_mean_squared_error_weighted(weight=0.0):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.exp(tf.multiply(weight,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_squared_error_weighted_linear(weight=0.0):
    # weight here is actually slope
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.add(1.0,tf.multiply(weight,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_squared_error_weighted_gaussian(weight=0.0):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.exp(tf.multiply(weight,tf.square(y_true))) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_squared_error_weighted_genexp(weight=(1.0,0.0,0.0)):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( \
            tf.multiply( weight[0], tf.exp( tf.multiply( weight[1], tf.pow(y_true,weight[2]) ) ) ) , \
            tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_absolute_error_weighted_genexp(weight=(1.0,0.0,0.0)):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( \
            tf.multiply( weight[0], tf.exp( tf.multiply( weight[1], tf.pow(y_true,weight[2]) ) ) ) , \
            tf.abs(tf.subtract(y_pred,y_true)) ) )
    return loss

################################################################
# Custom categorical metrics
# Note: categorical metrics assume y_true 0-1 scaling maps to 0-60 dBZ

thr_20dbz = 0.333
thr_35dbz = 0.583
thr_50dbz = 0.833

def my_csi20_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_20dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_20dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return nhit / (nhit + nmis + nfal)

def my_csi35_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_35dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_35dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return nhit / (nhit + nmis + nfal)
    
def my_csi50_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_50dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_50dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return nhit / (nhit + nmis + nfal)

def my_bias20_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_20dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_20dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return (nhit + nfal) / (nhit + nmis)

def my_bias35_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_35dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_35dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return (nhit + nfal) / (nhit + nmis)
    
def my_bias50_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_50dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_50dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return (nhit + nfal) / (nhit + nmis)

################################################################

