import os, numpy as np
from enum import Enum
import random, scipy.io, scipy.misc, imageio, math
from PIL import Image

''' Common interface for data reader.
'''
class DataInterface(object):
    
    ''' Reset to read from first.
    '''
    def reset():
        raise NotImplementedError()
    
    ''' Read data with the number of specific size.
    
    Args:
        size: Batch size.
    
    Returns:
        Tuple of RGB images, heatmaps, metric threshold and masks.
        RGB images are shape of (size, 256, 256, 3).
        Heatmaps are shape of (size, 64, 64, joint).
        Metric threshold is a single value.
        Masks are shape of (joint).
    '''
    def getBatch(size):
        raise NotImplementedError()

''' Common interface to combine data, task and metric.
'''
class DataCenter(object):
    
    ''' Set path to data.
    
    Args:
        root: relative path to root of data.
    '''
    def __init__(self, root):
        self.__root = root
    
    ''' Combine data, task and metric.
    
    Args:
        data: 'MPII' or 'FLIC'.
        task: 'train' or 'eval'.
        metric: 'PCK' or 'PCKh'.
    '''
    def request(self, data, task, metric):
        path = os.path.join(self.__root, data)
        
        if data == 'FLIC':
            return FLIC(root = path, task = task, metric = metric)
        elif data == 'MPII':
            return MPII(root = path, task = task, metric = metric)
        else:
            raise NotImplementedError()

''' Indices for joints.

FLIC and MPII joints are mixed.
'''
class JOINT(Enum):

    # FLIC
    L_Shoulder  =   0
    R_Shoulder  =   1
    L_Elbow     =   2
    R_Elbow     =   3
    L_Wrist     =   4
    R_Wrist     =   5
    L_Hip       =   6
    R_Hip       =   7
    L_Knee      =   8
    R_Knee      =   9
    L_Ankle     =   10
    R_Ankle     =   11

    L_Eye       =   12
    R_Eye       =   13
    L_Ear       =   14
    R_Ear       =   15
    M_Nose      =   16

    M_Shoulder  =   17
    M_Hip       =   18
    M_Ear       =   19
    M_Torso     =   20
    M_LUpperArm =   21
    M_RUpperArm =   22
    M_LLowerArm =   23
    M_RLowerArm =   24
    M_LUpperLeg =   25
    M_RUpperLeg =   26
    M_LLowerLeg =   27
    M_RLowerLeg =   28

    # MPII
    M_Pelvis    =   29
    M_Thorax    =   30
    M_UpperNeck =   31
    M_HeadTop   =   32


''' Data augmentation.
'''
class AUGMENTATION:
    class ROTATE(Enum):
        NO_CHANGE = 0
        CW_30_DEGREES = 1
        CCW_30_DEGREES = 2
    
    class SCALE(Enum):
        NO_CHANGE = 0
        UP_25_PERCENTAGE = 1
        DOWN_25_PERCENTAGE = 2


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
    if rotate == AUGMENTATION.ROTATE.CW_30_DEGREES.value:
        image = scipy.misc.imrotate(image, 30)
    elif rotate == AUGMENTATION.ROTATE.CCW_30_DEGREES.value:
        image = scipy.misc.imrotate(image, -30)

    if scale == AUGMENTATION.SCALE.UP_25_PERCENTAGE.value:
        new_length = int(256 * 1.25)
        new_center = int(256//2 * 1.25)
        image = scipy.misc.imresize(image, (new_length, new_length))
        image = image[
            new_center - 256//2 : new_center + 256//2,
            new_center - 256//2 : new_center + 256//2,
            0:3
        ]
    elif scale == AUGMENTATION.SCALE.DOWN_25_PERCENTAGE.value:
        new_length = int(256 * 0.75)
        image = scipy.misc.imresize(image, (new_length, new_length))
        image = np.pad(
            image,
            (
                (int(256 * 0.25/2), int(256 * 0.25/2)),
                (int(256 * 0.25/2), int(256 * 0.25/2)),
                (0, 0)
            ),
            'constant', constant_values = (0, 0)
        )
    return image


def transformPosition(pose, rotate, scale):    
    if rotate is not AUGMENTATION.ROTATE.NO_CHANGE.value:
        if rotate == AUGMENTATION.ROTATE.CW_30_DEGREES.value:
            degree = 30 
        elif rotate == AUGMENTATION.ROTATE.CCW_30_DEGREES.value:
            degree = -30

        pose_in_crop_space = {
            'vertical': pose['vertical'] - 64//2,
            'horizontal': pose['horizontal'] - 64//2
        }
        rotated_in_crop_space = {
            'vertical': math.cos(degree * math.pi / 180) * pose_in_crop_space['vertical']\
                - math.sin(degree * math.pi / 180) * pose_in_crop_space['horizontal'],
            'horizontal': math.sin(degree * math.pi / 180) * pose_in_crop_space['vertical']\
                + math.cos(degree * math.pi / 180) * pose_in_crop_space['horizontal']
        }
        pose = {
            'vertical': rotated_in_crop_space['vertical'] + 64//2,
            'horizontal': rotated_in_crop_space['horizontal'] + 64//2
        }

    if scale is not AUGMENTATION.SCALE.NO_CHANGE.value:
        if scale == AUGMENTATION.SCALE.UP_25_PERCENTAGE.value:
            coefficient = 1.25
        elif scale == AUGMENTATION.SCALE.DOWN_25_PERCENTAGE.value:
            coefficient = 0.75

        pose_in_crop_space = {
            'vertical': pose['vertical'] - 64//2,
            'horizontal': pose['horizontal'] - 64//2
        }
        scaled_in_crop_space = {
            'vertical': pose_in_crop_space['vertical'] * coefficient,
            'horizontal': pose_in_crop_space['horizontal'] * coefficient
        }
        pose = {
            'vertical': scaled_in_crop_space['vertical'] + 64//2,
            'horizontal': scaled_in_crop_space['horizontal'] + 64//2
        }
    
    return pose


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
        'constant', constant_values = (0, 0)
    )
    image = image[
        center['vertical'] + pad['vertical']['up'] - length//2
            : center['vertical'] + pad['vertical']['up'] + length//2,
        center['horizontal'] + pad['horizontal']['left'] - length//2
            : center['horizontal'] + pad['horizontal']['left'] + length//2,
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
def generateHeatmap(size, sigma = 1, center = None):
    x = np.arange(0, size, 1, np.float32)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[1]
        y0 = center[0]

    return 255/sigma*np.exp(-((x-x0)**2 + (y-y0)**2) / 2*sigma**2)


import sys
script_path = os.path.split(os.path.realpath(__file__))[0]
implementation_path = os.path.join(script_path, 'data')
sys.path.append(implementation_path)
                  
from FLIC import *
from MPII import *