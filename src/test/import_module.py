from src.module.bottleneck import bottleneck
import tensorflow as tf

with tf.variable_scope('bottleneck'):
    shape = [32, 64, 64, 256]
    train = tf.constant(value=False, dtype=tf.bool)
    x = tf.get_variable(
        name='input',
        dtype=tf.float32,
        shape=shape)
    assert bottleneck(input=x, train=train).get_shape() == shape

print('complete.')
