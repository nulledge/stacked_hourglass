import tensorflow as tf

import module
import module.layers as layers


class Model:
    def __init__(self, stages, features, joints):
        self.stages = stages
        self.features = features
        self.joints = joints

        with tf.variable_scope('input'):
            self.images = tf.placeholder(
                name='image',
                dtype=tf.float32,
                shape=[None, 256, 256, 3])
            self.heatmaps_groundtruth = tf.placeholder(
                name='heatmap_groundtruth',
                dtype=tf.float32,
                shape=[None, 64, 64, self.joints])
            self.is_training = tf.placeholder(
                name='train',
                dtype=tf.bool,
                shape=())
            self.mask = tf.placeholder(
                name='mask',
                dtype=tf.float32,
                shape=[None, self.joints]
            )

        with tf.variable_scope('compress'):
            with tf.variable_scope('conv_bn_relu'):
                net = layers.conv(inputs=self.images, ksize=7, kchannel=64, kstride=2)  # 128 * 128 * 64
                net = layers.bn(inputs=net, is_training=self.is_training)
                net = layers.relu(inputs=net)

            net = module.bottleneck(inputs=net, kchannel=128, is_training=self.is_training, name='A')  # 128 * 128 * 128
            net = layers.pool(inputs=net)  # 64 * 64 * 128
            net = module.bottleneck(inputs=net, kchannel=128, is_training=self.is_training, name='B')  # 64 * 64 * 128
            net = module.bottleneck(inputs=net, kchannel=self.features, is_training=self.is_training, name='C')  # 64 * 64 * 256

        self.heatmaps = []
        self.outputs = []
        for stage in range(1, stages + 1):
            with tf.variable_scope('hourglass_' + str(stage)):
                prev = tf.identity(net)
                net = module.hourglass(inputs=net, is_training=self.is_training)  # 64 * 64 * 256

                with tf.variable_scope('inter_hourglasses'):
                    net = module.bottleneck(inputs=net, is_training=self.is_training)  # 64 * 64 * 256
                    net = layers.conv(inputs=net, ksize=1, kchannel=self.features)  # 64 * 64 * 256
                    net = layers.bn(inputs=net, is_training=self.is_training)
                    net = layers.relu(inputs=net)

                with tf.variable_scope('heatmap'):
                    heatmap = layers.conv(inputs=net, ksize=1, kchannel=self.joints)  # 64 * 64 * joint

                    self.heatmaps.append(heatmap)
                    self.outputs.append(layers.merge(self.images, heatmap))

                net = layers.conv(inputs=net, ksize=1, kchannel=self.features, name='inter') \
                      + layers.conv(inputs=heatmap, ksize=1, kchannel=self.features, name='heatmap') \
                      + prev  # 64 * 64 * 256

        tf.summary.image('result-images', self.outputs[-1])
