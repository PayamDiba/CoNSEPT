"""
@author: Payam Dibaeinia
"""
"""
Every change from Run13 implementation was marked by #changed
"""



import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import tensorflow.keras.layers as layers

class CNN_fc(Model):
    """
    This CNN model uses a fully connected layer at the end to output expression
    """
    def __init__(self,
                 dropout_rate,
                 poolSize_bind,
                 convSize_coop,
                 stride_coop,
                 coopAct,
                 coop_inds,
                 fcConvChan_coop,
                 outAct,
                 cl2,
                 fcl2,
                 noPriorChan,
                 nTF):
        """
        coopAct: string
        outAct: string
        fcConvChan_coop: a list containing the number of filters (channels) for each convolutional layer after the very first coop layer
        nCoop: Number of cooperative interactions
        """
        super(CNN_fc, self).__init__()
        #TODO: Is it possible to use tf.Sequential here?
        #self.bn_bind = layers.BatchNormalization()
        self.bn_bind = layers.LayerNormalization()
        self.maxPool_bind = layers.MaxPool2D(pool_size = poolSize_bind, strides = poolSize_bind)
        self.maxPool_coop = layers.AveragePooling2D(pool_size = (2,1), strides = (2,1))
        self.coop_inds = coop_inds
        self.noPriorChan = noPriorChan
        nCoop = len(coop_inds)

        # after stacking, expand the last dim
        # Input to below conv has size: #batch * L * 2 * 1 (last dimension comes from expanding)
        # self.conv11_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        # self.conv22_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        # self.conv33_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        # self.conv12_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        # self.conv13_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        # self.conv23_coop = layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal')
        self.convs_coop = [layers.Conv2D(filters = 1, kernel_size = convSize_coop, strides = stride_coop, use_bias = True, kernel_initializer='glorot_normal', kernel_regularizer = tf.keras.regularizers.L2(cl2)) \
                           for _ in range(nCoop)]
        # output from each layer above has size : #batch * L * 1 * 1
        # Concat them to get #batch * L * 1 * #priorCoop
        if noPriorChan > 0:
            self.convs_coop_noPrior = layers.Conv2D(filters = noPriorChan, kernel_size = (convSize_coop[0],nTF), strides = (stride_coop[0],nTF), use_bias = True, kernel_initializer='glorot_normal', kernel_regularizer = tf.keras.regularizers.L2(cl2))
        # output from layer above has size : #batch * L * 1 * noPriorChan
        # Concat it with previous ones to get #batch * L * 1 * (#priorCoop+#noPriorChan)
        self.coopAct = layers.Activation(coopAct)



        self.fcConv_coop = []
        for nFilters in fcConvChan_coop:
            # We could use Conv1D, instead we used Conv2D but will squeeze once at the end
            #self.fcConv_coop.append(layers.LayerNormalization()) #changed
            self.fcConv_coop.append(layers.Conv2D(filters = nFilters, kernel_size = (3,1), strides = 1, use_bias = True, kernel_initializer='glorot_normal', kernel_regularizer = tf.keras.regularizers.L2(fcl2)))
            self.fcConv_coop.append(layers.AveragePooling2D(pool_size = (2,1), strides = (2,1)))
            self.fcConv_coop.append(layers.LayerNormalization())
            self.fcConv_coop.append(layers.Activation(coopAct))
            self.fcConv_coop.append(layers.Dropout(rate = dropout_rate))
            #self.fcConv_coop.append(layers.LayerNormalization())

        ### remove the  last dropout layers:
        #self.fcConv_coop.pop(-4)
        self.fcConv_coop = self.fcConv_coop[:-1]
        #self.drop_coop = layers.Dropout(rate = dropout_rate)
        #self.drop_coop2 = layers.Dropout(rate = dropout_rate)
        #self.maxPool_Coop = layers.MaxPool2D(pool_size = (3,1), strides = (3,1))
        self.flatten = layers.Flatten() #Changed from the very original implementation
        self.fc = layers.Dense(1, kernel_initializer='glorot_normal')

        if outAct:
            self.outAct = layers.Activation(outAct)
        else:
            self.outAct = None



    def call(self, inputs, training = True):
        """
        coop_inds: 2d array, each row shows pair of interacting TF indices
        # TODO: allow coops = None (default)
        """
        seq, conc = inputs
        nTF = conc.shape[1]

        seq = self.bn_bind(seq, training = training)
        seq = self.maxPool_bind(seq)
        seq = tf.squeeze(seq, axis = 2)

        conc = tf.tile(conc, [1,seq.shape[1]])
        conc = tf.reshape(conc, (-1, seq.shape[1], nTF))
        seq = tf.multiply(seq, conc)


        coop_list = []
        for k, idx in zip (self.convs_coop, self.coop_inds):
            c_ij = tf.stack((seq[:,:,idx[0]], seq[:,:,idx[1]]), axis = 2)
            c_ij = tf.expand_dims(c_ij, -1)
            coop_list.append(k(c_ij))

        seq_coop_noPrior = tf.expand_dims(seq,-1)
        if self.noPriorChan > 0:
            coop_noPrior = self.convs_coop_noPrior(seq_coop_noPrior)
            coop_list.append(coop_noPrior)


        coop = tf.concat(coop_list, axis = -1)
        coop = self.maxPool_coop(coop)
        #coop = self.bn_bind(coop, training = training)
        coop = self.coopAct(coop)
        #coop = self.drop_coop1(coop, training = training)
        for coop_layer in self.fcConv_coop:
            #if isinstance(coop_layer, layers.BatchNormalization):
            if isinstance(coop_layer, layers.LayerNormalization) or isinstance(coop_layer, layers.Dropout):
                coop = coop_layer(coop, training = training)
            else:
                coop = coop_layer(coop)
        #coop = self.drop_coop2(coop, training = training)
        #if coop.shape[1] > 2:
        #    coop = self.maxPool_Coop(coop)
        coop = tf.squeeze(coop, axis = 2)
        coop = self.flatten(coop)
        #print(coop.shape)
        ret = self.fc(coop)
        if self.outAct:
            ret = self.outAct(ret)

        #if self.outAct == 'tanh':
        #    ret = 0.5 * (ret + 1)

        return ret
