import tensorflow as tf


def bn(inputs, is_training):
    """Wrapper of batch norm.

    Args:
        inputs: The input tensor.
        is_training: The flag for training mode.

    Returns:
        tf.contrib.layers.batch_norm tensor.
    """
    # return tf.contrib.layers.batch_norm(
    #     inputs=inputs,
    #     is_training=is_training)
    return tf.layers.batch_normalization(inputs, training=is_training)
