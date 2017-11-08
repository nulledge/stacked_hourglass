import sys
sys.path.append('../')

import module
import tensorflow as tf

with tf.variable_scope('bottleneck'):
    shape = [32, 64, 64, 256]
    input = tf.get_variable(
        name = 'input',
        dtype = tf.float32,
        shape = shape)
    assert module.bottleneck(input = input, train = False).get_shape() == shape

print('complete.')