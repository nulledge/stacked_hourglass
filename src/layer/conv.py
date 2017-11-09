import tensorflow as tf

def conv(tensor, ksize, kchannel, kstride = 1):
    '''2D convolution with activation.
    
    Args:
        tensor: An input tensor in NHWC.
        ksize: The size of filter.
        kchannel: The number of filter channel.
        kstride: The stride interval.

    Returns:
        The output tensor of conv with activation.
    '''
    filter = [ksize, ksize, tensor.get_shape()[-1], kchannel]
    stride = [1, kstride, kstride, 1]

    kernel = tf.get_variable(
        name = 'filter',
        shape = filter,
        dtype = tf.float32)
    bias = tf.get_variable(
        name = 'bias',
        shape = [kchannel],
        dtype = tf.float32)
    conv = tf.nn.conv2d(
        name = 'conv',
        input = tensor,
        filter = kernel,
        strides = stride,
        padding = 'SAME',
        data_format = 'NHWC') + bias
    return conv