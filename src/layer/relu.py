import tensorflow as tf

def relu(input, name=None):
    '''Wrapper of ReLU.

    Args:
        input: The input tensor.
        name: The name of tensor.

    Returns:
        tf.nn.relu tensor.
    '''
    return tf.nn.relu(
        features = input,
        name = name)