import tensorflow as tf

from . import layers
from ._bottleneck import bottleneck


def hourglass(inputs, is_training):
    """Hourglass module.

    Args:
        inputs: An input tensor.
        is_training: The flag for batch normalization.

    Returns:
        An hourglass module tensor.
    """

    return tf.identity(name='output',
                       input=__hourglass_internal(inputs=inputs, is_training=is_training, depth=3))


def __hourglass_internal(inputs, is_training, depth):
    assert depth >= 0

    scope = lambda name: 'depth_' + str(depth) + '/' + name

    skip = bottleneck(inputs=inputs, is_training=is_training, name=scope('skip'))
    net = layers.pool(inputs=inputs, name=scope('pool'))
    net = bottleneck(inputs=net, is_training=is_training, name=scope('conv_downscale'))

    if depth == 0:
        net = bottleneck(inputs=net, is_training=is_training, name=scope('conv_center'))
    else:
        net = __hourglass_internal(inputs=net, is_training=is_training, depth=depth - 1)

    net = bottleneck(inputs=net, is_training=is_training, name=scope('conv_upscale'))

    height, width = map(lambda i: i.value * 2, net.get_shape()[-3:-1])
    net = tf.image.resize_images(
        images=net,
        size=[height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.add(net, skip, name=scope('output'))
