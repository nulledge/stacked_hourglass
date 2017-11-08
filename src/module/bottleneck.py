import tensorflow as tf

def bottleneck(input, train):
    '''Bottleneck module for residual networks with BN before convolution.

    Args:
        input: An input tensor in NHWC. The number of channel must be 256.
        train: The flag for batch normalization.

    Returns:
        The output tensor of the module.
    '''

    class Filter:
        pass
    filter = Filter()
    filter.size = [1, 3, 1]
    filter.channel = [64, 64, 256, input.get_shape()[-1]]
    filter.shape = [
        [filter.size[0], filter.size[0], filter.channel[-1], filter.channel[0]],
        [filter.size[1], filter.size[1], filter.channel[ 0], filter.channel[1]],
        [filter.size[2], filter.size[2], filter.channel[ 1], filter.channel[2]]
    ]

    assert filter.channel[-1] == filter.channel[2]

    def conv(tensor, index, activation = None, filter = filter):
        '''2D convolution with activation.
        
        Args:
            tensor: An input tensor in NHWC.
            index: The index of filter in 0-based index.
            activation: The activation function after convolution. If None then
                no activation.
            filter: The metadata of filters.

        Returns:
            The output tensor of conv with activation.
        '''
        assert index in range(0, 3)
        kernel = tf.get_variable(
            name = 'filter',
            shape = filter.shape[index],
            dtype = tf.float32)
        bias = tf.get_variable(
            name = 'bias',
            shape = [filter.channel[index]],
            dtype = tf.float32)
        conv = tf.nn.conv2d(
            name = 'conv',
            input = tensor,
            filter = kernel,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
            data_format = 'NHWC') + bias
        if activation is None:
            return conv
        else:
            return activation(conv)

    skip_connection = tf.identity(input)

    input = tf.contrib.layers.batch_norm(inputs = input, is_training = train)

    with tf.variable_scope('conv_00'):
        conv_00 = conv(tensor = input, index = 0, activation = tf.nn.relu)
        conv_00 = tf.contrib.layers.batch_norm(inputs = conv_00, is_training = train)

    with tf.variable_scope('conv_01'):
        conv_01 = conv(tensor = conv_00, index = 1, activation = tf.nn.relu)
        conv_01 = tf.contrib.layers.batch_norm(inputs = conv_01, is_training = train)

    with tf.variable_scope('conv_02'):
        conv_02 = conv(tensor = conv_01, index = 2, activation = None)

    output = conv_02

    return tf.nn.relu(output + skip_connection)