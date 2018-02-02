import tensorflow as tf


def pool(inputs, name=None):
    '''Wrapper of max pooling.

    Args:
        inputs: The input tensor.
        name: The name of tensor.

    Returns:
        tf.nn.max_pool tensor.
    '''
    return tf.nn.max_pool(
        value=inputs,
        name=name,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
