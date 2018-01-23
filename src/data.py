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