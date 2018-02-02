""" Import dependencies.

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
"""
import datetime
import glob
import json
import logging.handlers
import os

import numpy as np
import tensorflow as tf
from dotmap import DotMap
from tqdm import tqdm as tqdm

from datainfo import JOINT, getReader
# set logger
from model import Model

logger = logging.getLogger('HG')
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('./myLoggerTest.log')
streamHandler = logging.StreamHandler()
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
fileHandler.setFormatter(fomatter)
streamHandler.setFormatter(fomatter)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

''' Parsing config.

The config file is saved to root/config.
'''
CONFIG_PATH = os.path.abspath('config.json')
with open(CONFIG_PATH) as CONFIG_JSON:
    CONFIG = DotMap(json.loads(CONFIG_JSON.read()))
MODEL = CONFIG.model
TENSORBOARD = CONFIG.tensorboard
DATASET = CONFIG.dataset


def training(sess, model):
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("start training sequence at %s." % start_time)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    with tf.variable_scope('loss'):
        loss = tf.losses.mean_squared_error(
            tf.reshape(model.heatmaps_groundtruth,
                       shape=[-1, 64 * 64, model.joints]),
            tf.reshape(model.heatmaps[0],
                       shape=[-1, 64 * 64, model.joints]) * tf.expand_dims(model.mask, axis=1))

        for stage in range(1, model.stages):
            loss = loss + tf.losses.mean_squared_error(
                tf.reshape(model.heatmaps_groundtruth,
                           shape=[-1, 64 * 64, model.joints]),
                tf.reshape(model.heatmaps[stage],
                           shape=[-1, 64 * 64, model.joints]) * tf.expand_dims(model.mask, axis=1))

        tf.summary.scalar('loss', loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(name='optimizer', learning_rate=0.001).minimize(loss,
                                                                                               global_step=global_step)

    summary_merged = tf.summary.merge_all()
    logger.info('created ops for training (loss and optimizer)')

    tb_path = os.path.join(TENSORBOARD.path, start_time)
    os.makedirs(tb_path)
    writer = tf.summary.FileWriter(tb_path, sess.graph)
    logger.info("set tensorboard logdir: %s" % tb_path)

    sess.run(tf.global_variables_initializer())
    logger.info('initialized all variables')

    saver = tf.train.Saver(max_to_keep=20)
    if MODEL.pretrained.is_using:
        list_of_files = glob.glob(os.path.join(MODEL.pretrained.path, '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        file_name = os.path.basename(latest_file).split('.ckpt')[0]
        saver.restore(sess, os.path.join(MODEL.pretrained.path, '%s.ckpt' % file_name))
        logger.info('load the pretrained model')

    reader = getReader(DATASET.path, DATASET.name, MODEL.batch_size)
    logger.info('create dataset reader')

    tf.train.global_step(sess, global_step)
    for epoch in range(1, MODEL.epoch + 1):
        logger.info('training: epoch %d' % epoch)
        with tqdm(total=len(reader), unit=' images') as pbar:
            for train_images, train_heatmaps, _, _, train_mask in reader:
                _, result, summary, step = sess.run([optimizer, loss, summary_merged, tf.train.get_global_step()],
                                                    feed_dict={
                                                        model.images: train_images,
                                                        model.heatmaps_groundtruth: train_heatmaps,
                                                        model.is_training: True,
                                                        model.mask: train_mask
                                                    })
                pbar.set_postfix(loss=result)
                pbar.set_description("epoch: %d/%d" % (epoch, MODEL.epoch))
                pbar.update(train_images.shape[0])
                writer.add_summary(summary, step)

        save_path = saver.save(sess,
                               os.path.join(MODEL.path, start_time, '%s_%03d.ckpt' % (DATASET.name, epoch)))
        logger.info('saved the model at %s.' % save_path)


def evalution(sess, model):
    flat = tf.reshape(model.heatmaps[-1], shape=[-1, 64 * 64, model.joints])
    pose = tf.argmax(flat, axis=-2)

    if DATASET.name == 'FLIC':
        one_epoch = FLIC.NUMBER_OF_DATA - int(FLIC.NUMBER_OF_DATA * FLIC.TRAIN_RATIO)
    elif DATASET.name == 'MPII':
        one_epoch = MPII.NUMBER_OF_DATA - int(MPII.NUMBER_OF_DATA * MPII.TRAIN_RATIO)

    reader = getReader(DATASET.path, DATASET.name)

    eval_iter = tqdm(total=one_epoch, desc='ckpt: ')
    cnt = [0] * len(JOINT)
    for i in range(one_epoch):
        eval_images, eval_heatmaps, eval_pose, eval_threshold, eval_mask = reader.__getMiniBatch(8)
        if eval_images.shape[0] == 0:
            break
        pred, result = sess.run([pose, output],
                                feed_dict={
                                    model.images: eval_images,
                                    model.heatmaps_groundtruth: eval_heatmaps,
                                    model.is_training: False,
                                    model.mask: eval_mask})
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
                if dist <= eval_threshold[batch] * DATASET.metric.coefficient:
                    cnt[joint.value] += 1
    for joint in JOINT:
        if joint not in MPII.JOINT_TO_INDEX:
            continue
        print(joint, cnt[joint.value] / one_epoch * 100, end='%\n')

    eval_iter.close()


def main(_):
    num_stages = 8

    model = Model(num_stages, 128, len(JOINT))
    with tf.Session() as sess:

        if MODEL.task == 'train':
            training(sess, model)
        else:
            pass
            # evalution(sess, model)


if __name__ == "__main__":
    tf.app.run()
