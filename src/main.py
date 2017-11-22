import layer
import module
import tensorflow as tf
import os

class Flag:
    pass
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ckpt', 'hourglass.ckpt')
print('ckpt_path:', ckpt_path)
flag = Flag()
flag.train = False
flag.batch_size = 8

with tf.variable_scope('input'):
    images = tf.get_variable(
        name = 'image',
        dtype = tf.float32,
        shape = [flag.batch_size, 256, 256, 3])
    heatmaps_groundtruth = tf.get_variable(
        name = 'heatmap_groundtruth',
        dtype = tf.float32,
        shape = [flag.batch_size, 11, 256, 256])
    train = tf.get_variable(
        name = 'train',
        dtype = tf.bool,
        shape = [1])

# input size 256 * 256 * 3

with tf.variable_scope('compress'):
    with tf.variable_scope('conv_bn_relu'):
        net = layer.conv(input = images, ksize = 7, kchannel = 64, kstride = 2) # 128 * 128 * 64
        net = layer.bn(input = net, train = train)
        net = layer.relu(input = net)

    net = module.bottleneck(input = net, kchannel = 128, train = train, name = 'A') # 128 * 128 * 128
    net = layer.pool(input = net) # 64 * 64 * 128
    net = module.bottleneck(input = net, kchannel = 128, train = train, name = 'B') # 64 * 64 * 128
    net = module.bottleneck(input = net, kchannel = 256, train = train, name = 'C') # 64 * 64 * 256

last_stage = 2
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

with tf.variable_scope('loss'):
    loss = tf.losses.mean_squared_error(heatmaps_groundtruth, heatmaps[0])
    for stage in range(1, last_stage):
        loss = loss + tf.losses.mean_squared_error(heatmaps_groundtruth, heatmaps[stage])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(name = 'optimizer', learning_rate = 0.00025).minize(loss)

with tf.Session() as sess:
    saver = tf.train.Saver()

    if flag.train == True:
        sess.run(tf.global_variables_initializer())

        for epoch in range(10):
            for iterator in range(563):
                train_images, train_heatmaps = reader.batch(flag.batch_size)
                sess.run(optimizer,
                    feed_dict = {
                        images: train_image,
                        heatmaps_groundtruth: train_heatmaps,
                        train: False})

                result = sess.run(loss,
                    feed_dict = {
                        images: train_image,
                        heatmaps_groundtruth: train_heatmaps,
                        train: False})
                wrap = lambda label, value: label + '(' + str(value) + ')'
                print(wrap('epoch', epoch), wrap('iter', iterator), wrap('loss', result))                

        save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)
    else:
        saver.restore(sess, save_path)