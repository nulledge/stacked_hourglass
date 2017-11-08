import sys
sys.path.append('../')

import module
import tensorflow as tf

with tf.variable_scope('bottleneck'):
    shape = [32, 64, 64, 256]
    train = tf.constant(value = False, dtype = tf.bool)
    input = tf.get_variable(
        name = 'input',
        dtype = tf.float32,
        shape = shape)
    assert module.bottleneck(input = input, train = train).get_shape() == shape

print('complete.')