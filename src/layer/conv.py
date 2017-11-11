import tensorflow as tf

def conv(input, ksize, kchannel, kstride = 1, name = 'conv'):
    '''2D convolution.
    
    Args:
        input: An input tensor in NHWC.
        ksize: The size of filter.
        kchannel: The number of filter channel.
        kstride: The stride interval.

    Returns:
        The output tensor of conv.
    '''
    filter = [ksize, ksize, input.get_shape()[-1], kchannel]
    stride = [1, kstride, kstride, 1]

    with tf.variable_scope(name):
        kernel = tf.get_variable(
            name = 'filter',
            shape = filter,
            dtype = tf.float32)
        bias = tf.get_variable(
            name = 'bias',
            shape = [kchannel],
            dtype = tf.float32)
        conv = tf.nn.conv2d(
            name = 'output',
            input = input,
            filter = kernel,
            strides = stride,
            padding = 'SAME',
            data_format = 'NHWC') + bias
        return conv