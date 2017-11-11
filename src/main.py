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

train = False
# input size 256 * 256 * 3

with tf.variable_scope('compress'):
    with tf.variable_scope('conv_bn_relu'):
        net = layer.conv(input = input, ksize = 7, kchannel = 64, kstride = 2) # 128 * 128 * 64
        net = layer.bn(input = net, train = train)
        net = layer.relu(input = net)

    net = module.bottleneck(input = net, kchannel = 128, train = train, name = 'A') # 128 * 128 * 128
    net = layer.pool(input = net) # 64 * 64 * 128
    net = module.bottleneck(input = net, kchannel = 128, train = train, name = 'B') # 64 * 64 * 128
    net = module.bottleneck(input = net, kchannel = 256, train = train, name = 'C') # 64 * 64 * 256

last_stage = 8
heatmaps = []

for stage in range(1, last_stage+1):
    with tf.variable_scope('hourglass_' + str(stage)):
        prev = tf.identity(net)
        net = module.hourglass(input = net, train = train) # 64 * 64 * 256

        with tf.variable_scope('inter_hourglasses'):
            net = module.bottleneck(input = net, train = train) # 64 * 64 * 256
            net = layer.conv(input = net, ksize = 1, kchannel = 256) # 64 * 64 * 256
            net = layer.bn(input = net, train = train)
            net = layer.relu(input = net)

        with tf.variable_scope('heatmap'):
            heatmap = layer.conv(input = net, ksize = 1, kchannel = 11) # 64 * 64 * 11
            heatmaps.append(heatmap)

        if stage != last_stage:
            net = layer.conv(input = net, ksize = 1, kchannel = 256, name = 'inter')\
                + layer.conv(input = heatmap, ksize = 1, kchannel = 256, name = 'heatmap')\
                + prev # 64 * 64 * 256

print(heatmaps)