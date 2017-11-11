import tensorflow as tf

def bn(input, train):
    '''Wrapper of batch norm.

    Args:
        input: The input tensor.
        train: The flag for training mode.
    
    Returns:
        tf.contrib.layers.batch_norm tensor.
    '''
    return tf.contrib.layers.batch_norm(
        inputs = input,
        is_training = train)