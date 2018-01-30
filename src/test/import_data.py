import imageio
import os
from tqdm import tqdm
import tensorflow as tf

from src.dataset import DataCenter, MPII, FLIC
from src.dataset.joint import JOINT

with tf.variable_scope('input'):
    images = tf.placeholder(
        name='images',
        dtype=tf.float32,
        shape=[None, 256, 256, 3])
    heatmaps = tf.placeholder(
        name='heatmaps',
        dtype=tf.float32,
        shape=[None, 64, 64, len(JOINT)])


class tf_Spectrum:
    Color = tf.constant([
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0]
    ], dtype=tf.float32)


def tf_gray2color(gray, spectrum=tf_Spectrum.Color):
    indices = tf.floor_div(gray, 64)

    t = tf.expand_dims((gray - indices * 64) / (64), axis=-1)
    indices = tf.cast(indices, dtype=tf.int32)

    return tf.add(
        tf.multiply(tf.gather(spectrum, indices), 1 - t),
        tf.multiply(tf.gather(spectrum, indices + 1), t)
    )


def tf_merge(rgb, heat):
    heat = tf.image.resize_images(
        heat,
        [256, 256]
    )
    heat = tf.reduce_max(
        heat,
        axis=-1
    )
    return tf.cast(
        tf.add(
            tf.multiply(
                tf_gray2color(heat),
                0.6
            ),
            tf.multiply(rgb, 0.4)
        ),
        dtype=tf.uint8
    )


overlayed = tf_merge(images, heatmaps)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reader = DataCenter(root=os.path.expanduser('~/Workspace/data/')) \
        .request(data='MPII', task='train', metric='PCKh')

    one_epoch = int(MPII.NUMBER_OF_DATA * MPII.TRAIN_RATIO) * 2
    cnt = 0
    iter = tqdm(total=one_epoch)
    for _ in range(1):
        gt_image, gt_heatmap, _, _, _= reader.getBatch(8)

        returned = gt_image.shape[0]

        if returned == 0:
            break

        result = sess.run(
            overlayed,
            feed_dict={
                images: gt_image,
                heatmaps: gt_heatmap
            }
        )

        for idx in range(returned):
            imageio.imwrite('../img/' + str(cnt + idx + 1) + '.jpg', result[idx])

        iter.update(returned)
        cnt += returned
    iter.close()