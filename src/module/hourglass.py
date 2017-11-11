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

        scope = lambda name: 'depth_' + str(depth) + '/' + name

        skip = bottleneck(input = input, train = train, name = scope('skip'))
        net = layer.pool(input = input, name = scope('pool'))
        net = bottleneck(input = net, train = train, name = scope('conv_downscale'))

        if depth == 0:
            net = bottleneck(input = net, train = train, name = scope('conv_center'))
        else:
            net = _hourglass_internal(input = net, train = train, depth = depth - 1)

        net = bottleneck(input = net, train = train, name = scope('conv_upscale'))
        
        height, width = map(lambda i: i.value*2, net.get_shape()[-3:-1])
        net = tf.image.resize_images(
            images = net,
            size = [height, width],
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return tf.add(net, skip, name = scope('output'))

    return tf.identity(name = 'output',
        input = _hourglass_internal(input = input, train = train, depth = 3))