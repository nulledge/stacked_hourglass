import imageio
import tensorflow as tf
from tqdm import tqdm

from datainfo import JOINT, getReader
from module.layers import tf_utils

with tf.variable_scope('input'):
    images = tf.placeholder(
        name='images',
        dtype=tf.float32,
        shape=[None, 256, 256, 3])
    heatmaps = tf.placeholder(
        name='heatmaps',
        dtype=tf.float32,
        shape=[None, 64, 64, len(JOINT)])

overlayed = tf_utils.merge(images, heatmaps)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reader = getReader('../dataset-files', 'MPII', 8)

    cnt = 0
    with tqdm(total=len(reader)) as pbar:
        for gt_image, gt_heatmap, _, _, _ in reader:
            # for _ in range(1):
            #     gt_image, gt_heatmap, _, _, _ = reader.__next__()

            returned = gt_image.shape[0]
            result = sess.run(
                overlayed,
                feed_dict={
                    images: gt_image,
                    heatmaps: gt_heatmap
                }
            )

            for idx in range(returned):
                # imageio.imwrite('../img/gt-%d.jpg' % cnt, gt_image[idx])
                imageio.imwrite('/media/grayish/Backup/tb-log/ht-%d.jpg' % cnt, result[idx])
                cnt += 1
            pbar.update(returned)
