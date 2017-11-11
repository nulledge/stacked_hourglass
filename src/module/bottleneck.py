import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]

src_path = os.path.join(script_path, '../')
sys.path.append(src_path)

import layer

import tensorflow as tf

def bottleneck(input, train, kchannel = None, name = 'bottleneck'):
    '''Bottleneck module for residual networks with BN before convolution.

    Args:
        input: An input tensor in NHWC.
        train: The flag for batch normalization.
        name: The name of variable scope.
        kchannel: The number of filter channels. If None then same as input.

    Returns:
        The output tensor of the module.
    '''
    with tf.variable_scope(name):

        if kchannel == None:
            skip = tf.identity(input)
            kchannel = input.get_shape()[-1].value
        else:
            skip = layer.conv(input = input, ksize = 1, kchannel = kchannel)

        with tf.variable_scope('conv_00'):
            net = layer.bn(input = input, train = train)
            net = layer.relu(input = net)
            net = layer.conv(input = net, ksize = 1, kchannel = kchannel/2)

        with tf.variable_scope('conv_01'):
            net = layer.bn(input = net, train = train)
            net = layer.relu(input = net)
            net = layer.conv(input = net, ksize = 3, kchannel = kchannel/2)

        with tf.variable_scope('conv_02'):
            net = layer.bn(input = net, train = train)
            net = layer.relu(input = net)
            net = layer.conv(input = net, ksize = 1, kchannel = kchannel)

        return layer.relu(net + skip, name = 'output')