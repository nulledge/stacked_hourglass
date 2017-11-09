import layer
import module
import tensorflow as tf

with tf.variable_scope('placeholder'):
    pass

input = tf.get_variable(
    name = 'input',
    dtype = tf.float32,
    shape = [32, 256, 256, 3]
)

# input size 256 * 256 * 3

with tf.variable_scope('compress'):
    net = layer.conv(tensor = input, ksize = 7, kchannel = 256, kstride = 2) # 128 * 128 * 256
    net = module.bottleneck(input = net, train = False) # 128 * 128 * 256
    net = tf.nn.max_pool(value = net, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') # 64 * 64 * 256

with tf.variable_scope('hourglass_00'):
    net = module.hourglass(input = net, train = False)
    
print(net)