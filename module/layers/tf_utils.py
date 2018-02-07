import tensorflow as tf


class Spectrum:
    Color = tf.constant([
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0]
    ], dtype=tf.float32)


def gray2color(gray, spectrum=Spectrum.Color):
    indices = tf.floor_div(gray, 64)

    t = tf.expand_dims((gray - indices * 64) / 64, axis=-1)
    indices = tf.cast(indices, dtype=tf.int32)

    return tf.add(
        tf.multiply(tf.gather(spectrum, indices), 1 - t),
        tf.multiply(tf.gather(spectrum, indices + 1), t)
    )


def merge(rgb, heat):
    rgb *= 255
    heat = tf.image.resize_images(
        heat * 255,
        [256, 256]
    )
    heat = tf.reduce_max(
        heat,
        axis=-1
    )
    return tf.cast(
        tf.add(
            tf.multiply(
                gray2color(heat),
                0.5
            ),
            tf.multiply(rgb, 0.5)
        ),
        dtype=tf.uint8
    )
