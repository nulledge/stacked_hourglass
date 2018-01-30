import numpy as np
import imageio
import math
import scipy.io
import scipy.misc


''' Transform the image.

Transform the image following the augmentation rules.

Args:
    image: Image with the shape of (256, 256, 3).
    rotate: One of AUGMENTATION.ROTATE.
    scale: One of AUGMENTATION.SCALE.

Return:
    Rotated and Scaled image with the shape of (256, 256, 3).
'''


def transformImage(image, rotate, scale):
    if scale > 1.0:
        new_length = int(256 * scale)
        new_center = int(256 // 2 * scale)
        image = scipy.misc.imresize(image, (new_length, new_length))
        image = scipy.misc.imrotate(image, rotate)
        image = image[
                new_center - 256 // 2: new_center + 256 // 2,
                new_center - 256 // 2: new_center + 256 // 2,
                0:3
                ]
    elif scale < 1.0:
        new_length = int(256 * scale)
        image = scipy.misc.imresize(image, (new_length, new_length))
        image = np.pad(
            image,
            (
                (int(256 * (1.0 - scale) / 2), int(256 * (1.0 - scale) / 2)),
                (int(256 * (1.0 - scale) / 2), int(256 * (1.0 - scale) / 2)),
                (0, 0)
            ),
            'constant', constant_values=(0, 0)
        )
        image = scipy.misc.imrotate(image, rotate)
        image = scipy.misc.imresize(image, (256, 256))
    return image


def transformPosition(pose, rotate, scale):
    if scale is not 1.0:
        pose_in_crop_space = {
            'vertical': pose['vertical'] - 64 // 2,
            'horizontal': pose['horizontal'] - 64 // 2
        }
        scaled_in_crop_space = {
            'vertical': pose_in_crop_space['vertical'] * scale,
            'horizontal': pose_in_crop_space['horizontal'] * scale
        }
        pose = {
            'vertical': scaled_in_crop_space['vertical'] + 64 // 2,
            'horizontal': scaled_in_crop_space['horizontal'] + 64 // 2
        }

    if rotate is not 0.0:
        pose_in_crop_space = {
            'vertical': pose['vertical'] - 64 // 2,
            'horizontal': pose['horizontal'] - 64 // 2
        }
        rotated_in_crop_space = {
            'vertical': math.cos(rotate * math.pi / 180) * pose_in_crop_space['vertical'] \
                        - math.sin(rotate * math.pi / 180) * pose_in_crop_space['horizontal'],
            'horizontal': math.sin(rotate * math.pi / 180) * pose_in_crop_space['vertical'] \
                          + math.cos(rotate * math.pi / 180) * pose_in_crop_space['horizontal']
        }
        pose = {
            'vertical': rotated_in_crop_space['vertical'] + 64 // 2,
            'horizontal': rotated_in_crop_space['horizontal'] + 64 // 2
        }

    outlier = int(pose['vertical']) not in range(0, 64)\
              or int(pose['horizontal']) not in range(0, 64)

    return pose, outlier


''' Crop RGB image

Args:
    image_path: Path to the RGB image.
    center: Center of the bounding box.
    length: Width and height of the bounding box.
    pad: Padding size to each sides.

Returns:

'''


def cropRGB(image_path, center, length, pad):
    image = imageio.imread(image_path)
    image = np.pad(
        image,
        (
            (pad['vertical']['up'], pad['vertical']['down']),
            (pad['horizontal']['left'], pad['horizontal']['right']),
            (0, 0)
        ),
        'constant', constant_values=(0, 0)
    )
    image = image[
            center['vertical'] + pad['vertical']['up'] - length // 2
            : center['vertical'] + pad['vertical']['up'] + length // 2,
            center['horizontal'] + pad['horizontal']['left'] - length // 2
            : center['horizontal'] + pad['horizontal']['left'] + length // 2,
            0:3
            ]
    image = scipy.misc.imresize(image, (256, 256))

    return image


''' Generate heatmap with Gaussian distribution.

Args:
    size: Width and height of heatmap.
    sigma: The variance of Gaussian distribution.
    center: The center of Gaussian distribution. Its shape must be (2).

Return:
    The heatmap of (64, 64) shape.
'''


def generateHeatmap(size, sigma=1, center=None):
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[1]
        y0 = center[0]

    return 255 / sigma * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2 * sigma ** 2)