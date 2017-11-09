import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]

src_path = os.path.join(script_path, '../')
sys.path.append(src_path)

import layer

import tensorflow as tf

def bottleneck(input, train, name = 'bottleneck'):
    '''Bottleneck module for residual networks with BN before convolution.

    Args:
        input: An input tensor in NHWC. The number of channel must be 256.
        train: The flag for batch normalization.
        name: The name of variable scope.

    Returns:
        The output tensor of the module.
    '''
    assert input.get_shape()[-1] == 256

    with tf.variable_scope(name):

        skip = tf.identity(input)

        net = tf.contrib.layers.batch_norm(inputs = input, is_training = train)
        with tf.variable_scope('conv_00'):
            net = layer.conv(tensor = net, ksize = 1, kchannel = 128)
            net = tf.contrib.layers.batch_norm(inputs = net, is_training = train)
            net = tf.nn.relu(net)

        with tf.variable_scope('conv_01'):
            net = layer.conv(tensor = net, ksize = 3, kchannel = 128)
            net = tf.contrib.layers.batch_norm(inputs = net, is_training = train)
            net = tf.nn.relu(net)

        with tf.variable_scope('conv_02'):
            net = layer.conv(tensor = net, ksize = 1, kchannel = 256)

        return tf.nn.relu(net + skip, name = 'output')