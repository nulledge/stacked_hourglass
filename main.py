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
            optimizer = tf.train.RMSPropOptimizer(name='optimizer', learning_rate=MODEL.lr).minimize(loss,
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

    reader = getReader(DATASET.path, DATASET.name, MODEL.batch_size, MODEL.task)
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
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("start evaluation sequence at %s." % start_time)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    poses_groundtruth = tf.placeholder(
        name='pose_groundtruth',
        dtype=tf.float32,
        shape=[None, model.joints, 2])
    thresholds_groundtruth = tf.placeholder(
        name='threshold_groundtruth',
        dtype=tf.float32,
        shape=[None])
    flat = tf.reshape(model.heatmaps[-1], shape=[-1, 64 * 64, model.joints])
    pose = tf.argmax(flat, axis=-2)
    pred = tf.stack(values=[pose // 64, pose % 64], axis=-1)
    dist = tf.norm(tf.cast(pred, tf.float32) - poses_groundtruth, axis=-1)
    thresholds_gt_exdim = tf.expand_dims(thresholds_groundtruth * DATASET.metric.coefficient, axis=-1)
    pred_hit = tf.less(dist, thresholds_gt_exdim)

    logger.info('created ops for evaluation.')

    sess.run(tf.global_variables_initializer())
    logger.info('initialized all variables')

    saver = tf.train.Saver(max_to_keep=20)
    if MODEL.pretrained.is_using:
        list_of_files = glob.glob(os.path.join(MODEL.pretrained.path, '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        file_name = os.path.basename(latest_file).split('.ckpt')[0]
        saver.restore(sess, os.path.join(MODEL.pretrained.path, '%s.ckpt' % file_name))
        logger.info('load the pretrained model')

    reader = getReader(DATASET.path, DATASET.name, MODEL.batch_size, MODEL.task)
    logger.info('create dataset reader.')

    tf.train.global_step(sess, global_step)
    hit = np.zeros(shape=model.joints, dtype=np.float)
    total = np.zeros(shape=model.joints, dtype=np.float)
    with tqdm(total=len(reader), unit=' images') as pbar:
        for eval_images, eval_heatmaps, eval_poses, eval_thresholds, eval_masks in reader:
            is_hit, step = sess.run([pred_hit, tf.train.get_global_step()],
                                    feed_dict={
                                        model.images: eval_images,
                                        model.heatmaps_groundtruth: eval_heatmaps,
                                        model.is_training: False,
                                        model.mask: eval_masks,
                                        poses_groundtruth: eval_poses,
                                        thresholds_groundtruth: eval_thresholds
                                    })
            total += np.count_nonzero(eval_masks, axis=0)
            hit += np.count_nonzero(is_hit * eval_masks, axis=0)
            pbar.update(eval_images.shape[0])

    logger.info("step(%d), total_acc(%f%%)" % (step, sum(hit) / sum(total) * 100))
    for joint in np.nonzero(total)[0]:
        in_percentage = hit[joint] / total[joint] * 100
        logger.info("%s: %f%% (%d/%d)" % (JOINT(joint).name, in_percentage, hit[joint], total[joint]))


def main(_):
    num_stages = 8

    model = Model(num_stages, MODEL.features, len(JOINT))
    with tf.Session() as sess:

        if MODEL.task == 'train':
            training(sess, model)
        else:
            evalution(sess, model)


if __name__ == "__main__":
    tf.app.run()
