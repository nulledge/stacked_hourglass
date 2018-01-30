'''from . import DataCenter
from .FLIC import FLIC
from .MPII import MPII

import numpy as np


class MIX:
    def __init__(self, root, task, metric):
        self.__root = root
        self.__task = task
        self.__metric = metric

        self.__FLIC = DataCenter(root=root).request(data='FLIC', task=task, metric='PCK')
        self.__MPII = DataCenter(root=root).request(data='MPII', task=task, metric='PCKh')

    def __delete__(self):
        pass

    def reset(self):
        self.__FLIC.reset()
        self.__MPII.reset()

    def getBatch(self, size):
        FLIC_ratio = FLIC.NUMBER_OF_DATA / (MPII.NUMBER_OF_DATA + FLIC.NUMBER_OF_DATA)
        FLIC_size = size * FLIC_ratio

        FLIC_rgb, FLIC_heat, FLIC_pose, FLIC_threshold, FLIC_ = self.__FLIC.getBatch(FLIC_size)
        batch_MPII = self.__MPII.getBatch(size - FLIC_size)

        FLIC_size = batch_FLIC[0].shape[0]
        MPII_size = batch_MPII[0].shape[0]
'''