import numpy as np
import logging

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np

def unet(x, keep_prob, channels, n_class, img_rows, img_cols, batch_norm, is_train, reuse, blocks=5, layers=2, features_root=32, filter_size=3, pool_size=2, pad="SAME", activation="elu", upsampling=1, initializer="Normal", summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param pad: size of padding
    :param summaries: Flag if summaries should be created (Not yet implemented)
    """
    # Summary for visualising intermediate tensors is not yet implemented

    # for changing activation function
    if activation == "relu":
        act = tf.nn.relu
    elif activation == "elu":
        act = tf.nn.elu
    elif activation == "leaky_relu":
        act = tf.nn.leaky_relu

    # activiation with/without batchnorm
    if batch_norm:
        act_bn = act
        act_conv = None
    else:
        act_conv = act
        act_bn = None

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Different initialization scheme for weights
    if initializer == "Normal":
        w_init = tf.truncated_normal_initializer(stddev=0.02)
    elif initializer == "He":
        w_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
    elif initializer == "Xavier":
        w_init = tf.contrib.layers.xavier_initializer()

    b_init = tf.constant_initializer(value=0.0)
    # initializer for beta in batchnorm
    beta_init = tf.zeros_initializer()
    # initializer for gamma in batchnorm
    gamma_init=tf.ones_initializer()
    
    ## Define the network architecture
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        data_tensor = InputLayer(x, name='inputs')
        store_tensor = {}

        # Downsampling leg and center block
        for b in range(blocks):
            for l in range(layers):
                data_tensor = Conv2d(data_tensor, features_root * (2 ** b), (filter_size, filter_size), padding=pad, W_init=w_init, b_init=b_init, act=act_conv, name='conv{0}_{1}'.format(str(b+1), str(l+1)))
                if batch_norm:
                    data_tensor = BatchNormLayer(data_tensor, act=act_bn, is_train=is_train, beta_init=beta_init, gamma_init=gamma_init, name='bn_{0}_{1}'.format(str(b+1), str(l+1)))


            # downsample if not last block, also store tensor for concat layer 
            if b != blocks - 1:
                store_tensor[str(b+1)] = data_tensor
                data_tensor = MaxPool2d(data_tensor, (pool_size, pool_size), name='pool_{}'.format(str(b+1)))
 
        # upsampling leg
        for b in range(blocks-1):

            # select one upsampling method
            if upsampling == 3:
                data_tensor = DeConv2d(data_tensor, features_root * (2 ** (blocks - 2 - b)), (filter_size, filter_size), (img_rows/(2 ** (blocks - 2 - b)), img_cols/(2 ** (blocks - 2 - b))), (pool_size, pool_size), name='Upsample_{0}'.format(b+1))
            else:
                data_tensor = UpSampling2dLayer(data_tensor, (pool_size,pool_size), is_scale=True, method=upsampling, name='Upsample_{0}'.format(b+1))

            data_tensor = ConcatLayer([data_tensor, store_tensor[str(blocks-1-b)]], 3, name='concat_{0}'.format(b+1))

            # convolution after upsampling
            for l in range(layers):

                data_tensor = Conv2d(data_tensor, features_root * (2 ** (blocks - 2 - b)), (filter_size, filter_size), padding=pad, W_init=w_init, b_init=b_init, act=act_conv, name='uconv{0}_{1}'.format(str(b+1),str(l+1)))
                if batch_norm:
                    data_tensor = BatchNormLayer(data_tensor, act=act_bn, is_train=is_train, beta_init=beta_init, gamma_init=gamma_init, name='ubn_{0}_{1}'.format(str(b+1), str(l+1)))

        # output 
        output_map = Conv2d(data_tensor, n_class, (1, 1), act=None, name='output')
        
    return output_map


