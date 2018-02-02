import tensorflow as tf


def relu(inputs, name=None):
    '''Wrapper of ReLU.

    Args:
        inputs: The input tensor.
        name: The name of tensor.

    Returns:
        tf.nn.relu tensor.
    '''
    return tf.nn.relu(
        features=inputs,
        name=name)
