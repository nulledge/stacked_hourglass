import tensorflow as tf

def pool(input, name=None):
    '''Wrapper of max pooling.

    Args:
        input: The input tensor.
        name: The name of tensor.

    Returns:
        tf.nn.max_pool tensor.
    '''
    return tf.nn.max_pool(
        value = input,
        name = name,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1], 
        padding = 'SAME')