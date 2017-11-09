from bottleneck import *
import tensorflow as tf

def hourglass(input, train):
    '''Hourglass module.

    Args:
        input: An input tensor.
        train: The flag for batch normalization.

    Returns:
        An hourglass module tensor.
    '''

    def _hourglass_internal(input, train, depth):
        assert depth >= 0

        with tf.variable_scope('depth_' + str(depth)):
            with tf.variable_scope('conv'):
                net = bottleneck(input = input, train = train)
            with tf.variable_scope('skip'):
                skip = bottleneck(input = net, train = train)
            net = tf.nn.max_pool(value = net, name = 'pool',
                ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

            if depth == 0:
                with tf.variable_scope('mini_conv_00'):
                    net = bottleneck(input = net, train = train)
                with tf.variable_scope('mini_conv_01'):
                    net = bottleneck(input = net, train = train)
                with tf.variable_scope('mini_conv_02'):
                    net = bottleneck(input = net, train = train)
            else:
                net = _hourglass_internal(input = net,
                    train = train, depth = depth - 1)

            print(net)

            height, width = map(lambda i: i.value*2, net.get_shape()[-3:-1])
            net = tf.image.resize_images(
                images = net,
                size = [height, width],
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
            return net + skip

    return _hourglass_internal(input = input, train = train, depth = 3)