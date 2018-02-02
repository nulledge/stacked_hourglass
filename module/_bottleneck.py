import tensorflow as tf

from . import layers


def bottleneck(inputs, is_training, kchannel=None, name='bottleneck'):
    '''Bottleneck module for residual networks with BN before convolution.

    Args:
        inputs: An input tensor in NHWC.
        is_training: The flag for batch normalization.
        name: The name of variable scope.
        kchannel: The number of filter channels. If None then same as input.

    Returns:
        The output tensor of the module.
    '''
    with tf.variable_scope(name):

        if kchannel == None:
            skip = tf.identity(inputs)
            kchannel = inputs.get_shape()[-1].value
            net = tf.identity(inputs)
        else:
            skip = layers.conv(inputs=inputs, ksize=1, kchannel=kchannel)
            net = layers.bn(inputs=inputs, is_training=is_training)
            net = layers.relu(inputs=net)

        with tf.variable_scope('conv_00'):
            net = layers.conv(inputs=net, ksize=1, kchannel=kchannel / 2)
            net = layers.bn(inputs=net, is_training=is_training)
            net = layers.relu(inputs=net)

        with tf.variable_scope('conv_01'):
            net = layers.conv(inputs=net, ksize=3, kchannel=kchannel / 2)
            net = layers.bn(inputs=net, is_training=is_training)
            net = layers.relu(inputs=net)

        with tf.variable_scope('conv_02'):
            net = layers.conv(inputs=net, ksize=1, kchannel=kchannel)

        return layers.relu(net + skip, name='output')
