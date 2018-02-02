import tensorflow as tf

import module

with tf.variable_scope('bottleneck'):
    shape = [32, 64, 64, 256]
    train = tf.constant(value=False, dtype=tf.bool)
    x = tf.get_variable(
        name='input',
        dtype=tf.float32,
        shape=shape)
    assert module.bottleneck(inputs=x, is_training=train).get_shape() == shape

print('complete.')
