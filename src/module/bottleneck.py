import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]

src_path = os.path.join(script_path, '../')
sys.path.append(src_path)

import layer

import tensorflow as tf

def bottleneck(input, train):
    '''Bottleneck module for residual networks with BN before convolution.

    Args:
        input: An input tensor in NHWC. The number of channel must be 256.
        train: The flag for batch normalization.

    Returns:
        The output tensor of the module.
    '''

    assert input.get_shape()[-1] == 256

    skip_connection = tf.identity(input)

    input = tf.contrib.layers.batch_norm(inputs = input, is_training = train)

    with tf.variable_scope('conv_00'):
        conv_00 = layer.conv(tensor = input, ksize = 1, kchannel = 128)
        conv_00 = tf.contrib.layers.batch_norm(inputs = conv_00, is_training = train)
        conv_00 = tf.nn.relu(conv_00)

    with tf.variable_scope('conv_01'):
        conv_01 = layer.conv(tensor = conv_00, ksize = 3, kchannel = 128)
        conv_01 = tf.contrib.layers.batch_norm(inputs = conv_01, is_training = train)
        conv_01 = tf.nn.relu(conv_01)

    with tf.variable_scope('conv_02'):
        conv_02 = layer.conv(tensor = conv_01, ksize = 1, kchannel = 256)

    output = conv_02

    return tf.nn.relu(output + skip_connection)