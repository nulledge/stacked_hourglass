
# coding: utf-8

# In[1]:


''' Import dependencies.

Moduels:
    tensorflow: The neural network framework.
    layer: The customized single layers.
    module: The customized multi-layer modules.
    
    os: The basic module for path parsing.
    json: The basic modue for config parsing.
    zipfile: Extract zip file.
    random: Shuffle indices.
    scipy.io: Load .mat file.
    numpy: To process .mat file data.
    
    tqdm: The 3rd-party looping visualzer.
'''
import tensorflow as tf
import datetime
import glob
import json

import os
import imageio
import numpy as np
from tqdm import tqdm as tqdm

from dataset import DataCenter
from dataset.FLIC import FLIC
from dataset.MPII import MPII
from dataset.joint import JOINT
import layer
import module

# In[ ]:


''' Parsing config.

The config file is saved to root/config.
'''
CONFIG_PATH = os.path.abspath('../config.json')
with open(CONFIG_PATH) as CONFIG_FILE:
    CONFIG = json.loads(CONFIG_FILE.read())
    flags = tf.app.flags
    flags.DEFINE_string('project', CONFIG['project'], 'The project name.')
    
    flags.DEFINE_string('path_to_pretrained',
                        os.path.join(
                            os.path.expanduser(CONFIG['pretrained']['path']),
                            CONFIG['project']),
                        'The path to pretrained parameters.')
    flags.DEFINE_boolean('load_pretrained', CONFIG['pretrained']['load'], 'Load the latest pretrained parameters.')
    flags.DEFINE_boolean('save_pretrained', CONFIG['pretrained']['save'], 'Save the trained parameters.')
    
    flags.DEFINE_string('path_to_log',
                        os.path.join(
                            os.path.expanduser(CONFIG['log']['path']),
                            CONFIG['project']),
                        'The path to log.')
    flags.DEFINE_boolean('save_log', CONFIG['log']['save'], 'Save the log.')
    
    flags.DEFINE_string('path_to_data',
                        os.path.expanduser(CONFIG['data']['path']),
                        'The path to data.')
    flags.DEFINE_string('name_of_data', CONFIG['data']['name'], 'The name of data.')
    
    flags.DEFINE_string('task', CONFIG['task']['step'], 'The task to be done.')
    flags.DEFINE_integer('epoch', CONFIG['task']['train']['epoch'], 'The epoch to be trained.')
    flags.DEFINE_string('metric', CONFIG['task']['eval']['metric'], 'The evaluation metric.')
    flags.DEFINE_float('metric_coefficient', CONFIG['task']['eval']['coefficient'], 'The evaluation metric coefficient.')
    
FLAGS = flags.FLAGS


# In[ ]:


with tf.variable_scope('input'):
    images = tf.placeholder(
        name = 'image',
        dtype = tf.float32,
        shape = [None, 256, 256, 3])
    heatmaps_groundtruth = tf.placeholder(
        name = 'heatmap_groundtruth',
        dtype = tf.float32,
        shape = [None, 64, 64, len(JOINT)])
    train = tf.placeholder(
        name = 'train',
        dtype = tf.bool,
        shape = ())
    mask = tf.placeholder(
        name = 'mask',
        dtype = tf.float32,
        shape = [None, len(JOINT)]
    )


# In[ ]:


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


# In[ ]:


class tf_Spectrum:
    Color = tf.constant([
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0]
    ], dtype = tf.float32)

def tf_gray2color(gray, spectrum = tf_Spectrum.Color):
    indices = tf.floor_div(gray, 64)
    
    t = tf.expand_dims((gray - indices * 64) / (64), axis = -1)
    indices = tf.cast(indices, dtype = tf.int32)
    
    return tf.add(
        tf.multiply(tf.gather(spectrum, indices), 1 - t),
        tf.multiply(tf.gather(spectrum, indices+1), t)
    )

def tf_merge(rgb, heat):
    heat = tf.image.resize_images(
        heat,
        [256, 256]
    )
    heat = tf.reduce_max(
        heat,
        axis = -1
    )
    return tf.cast(
        tf.add(
            tf.multiply(
                tf_gray2color(heat),
                0.6
            ),
            tf.multiply(rgb, 0.4)
        ),
        dtype = tf.uint8
    )


# In[ ]:


last_stage = 2
heatmaps = []
output = []

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
            heatmap = layer.conv(input = net, ksize = 1, kchannel = len(JOINT)) # 64 * 64 * joint
            
            heatmaps.append(heatmap)
            output.append(tf_merge(images, heatmap))

        if stage != last_stage:
            net = layer.conv(input = net, ksize = 1, kchannel = 256, name = 'inter')                + layer.conv(input = heatmap, ksize = 1, kchannel = 256, name = 'heatmap')                + prev # 64 * 64 * 256
summary_visualize = tf.summary.image('visualize', output[-1])

# In[ ]:


if FLAGS.task == 'train' :
    with tf.variable_scope('loss'):
        loss = tf.losses.mean_squared_error(
            tf.reshape(heatmaps_groundtruth, shape = [-1, 64*64, len(JOINT)]),
            tf.reshape(heatmaps[0], shape = [-1, 64*64, len(JOINT)]) * tf.expand_dims(mask, axis = 1))
        for stage in range(1, last_stage):
            loss = loss + tf.losses.mean_squared_error(
                tf.reshape(heatmaps_groundtruth, shape=[-1, 64 * 64, len(JOINT)]),
                tf.reshape(heatmaps[stage], shape=[-1, 64 * 64, len(JOINT)]) * tf.expand_dims(mask, axis=1))
        summary_loss = tf.summary.scalar('loss', loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(name = 'optimizer', learning_rate = 0.00025).minimize(loss)
            
if FLAGS.task == 'eval':
     pose = tf.argmax(
         tf.reshape(
             heatmaps[-1],
             shape = [-1, 64*64, len(JOINT)]
         ),
         axis = -2
     )


# In[ ]:


summary_merged = tf.summary.merge_all()

sess = tf.Session()
saver = tf.train.Saver()
writer = tf.summary.FileWriter(FLAGS.path_to_log, sess.graph)

reader = DataCenter(root = FLAGS.path_to_data).request(data = FLAGS.name_of_data, task = FLAGS.task, metric = FLAGS.metric)

if FLAGS.load_pretrained:
    list_of_files = glob.glob(os.path.join(FLAGS.path_to_pretrained, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    file_name = os.path.basename(latest_file).split('.ckpt')[0]
    saver.restore(sess, os.path.join(FLAGS.path_to_pretrained, file_name + '.ckpt'))
else:
    file_name = 'None'
    sess.run(tf.global_variables_initializer())


# In[ ]:


if FLAGS.task == 'train':
    idx = 0
    for epoch in range(1, FLAGS.epoch + 1):
        if FLAGS.name_of_data == 'FLIC':
            one_epoch = int(FLIC.NUMBER_OF_DATA * FLIC.TRAIN_RATIO) * 2
        elif FLAGS.name_of_data == 'MPII':
            one_epoch = int(MPII.NUMBER_OF_DATA * MPII.TRAIN_RATIO) * 2
        train_iter = tqdm(total = one_epoch, desc = 'epoch: ' + str(epoch) + '/' + str(FLAGS.epoch))
        reader.reset()
        for i in range(one_epoch):
            train_images, train_heatmaps, train_pose, train_threshold, train_mask = reader.getBatch(8)
            if train_images.shape[0] == 0:
                break
            _, result, summary = sess.run([optimizer, loss, summary_merged],
                feed_dict = {
                    images: train_images,
                    heatmaps_groundtruth: train_heatmaps,
                    train: True,
                    mask: train_mask
                })
            train_iter.set_postfix(loss = result)
            train_iter.update(train_images.shape[0])
            writer.add_summary(summary, idx)
            idx += 1
        train_iter.close();
        
        save_path = saver.save(sess, os.path.join(FLAGS.path_to_pretrained, str(datetime.datetime.now()) + '_MPII_' + str(epoch)  + '_FLIC_0.ckpt'))
        print('save to:', save_path)
else:
    if FLAGS.name_of_data == 'FLIC':
        one_epoch = FLIC.NUMBER_OF_DATA - int(FLIC.NUMBER_OF_DATA * FLIC.TRAIN_RATIO)
    elif FLAGS.name_of_data == 'MPII':
        one_epoch = MPII.NUMBER_OF_DATA - int(MPII.NUMBER_OF_DATA * MPII.TRAIN_RATIO)
        
    reader.reset()
    eval_iter = tqdm(total = one_epoch, desc = 'ckpt: ' + file_name)
    cnt = [0] * len(JOINT)
    for i in range(one_epoch):
        eval_images, eval_heatmaps, eval_pose, eval_threshold, eval_mask = reader.getBatch(8)
        if eval_images.shape[0] == 0:
            break
        pred, result = sess.run([pose, output],
            feed_dict = {
                images: eval_images,
                heatmaps_groundtruth: eval_heatmaps,
                train: False,
                mask: eval_mask})
        eval_iter.update(eval_images.shape[0])
        
        for batch in range(eval_images.shape[0]):                
            py = pred[batch] // 64
            px = pred[batch] % 64
            for joint in JOINT:
                if joint not in MPII.JOINT_TO_INDEX or eval_mask[batch][joint.value] == False:
                    continue
                dist = np.linalg.norm(
                    np.array([py[joint.value], px[joint.value]])
                    - np.array(eval_pose[batch][joint.value]))
                if dist <= eval_threshold[batch] * FLAGS.metric_coefficient:
                    cnt[joint.value] += 1
    for joint in JOINT:
        if joint not in MPII.JOINT_TO_INDEX:
            continue
        print(joint, cnt[joint.value] / one_epoch * 100, end = '%\n')
                
            
    eval_iter.close()


# reader.resetBatch()
# _, heat, _, _ = reader.getBatch(1)
# heat = heat[0]
# maximum = -1
# for y in range(64):
#     for x in range(64):
#         heat[y, x, 0] = max(heat[y, x, :])
#         if heat[y, x, 0] > maximum:
#             maximum = heat[y, x, 0]
# imageio.imwrite('img/heat.jpg', heat[:, :, 0])
# print(maximum)
