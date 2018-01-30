from .FLIC import FLIC
from .MPII import MPII
import random

import numpy as np
import os


class MIX:
    def __init__(self, root, task):
        assert task == 'train'

        FLIC_path = os.path.join(root, 'FLIC')
        MPII_path = os.path.join(root, 'MPII')

        self.__FLIC = FLIC(root=FLIC_path, task=task, metric='PCK')
        self.__MPII = MPII(root=MPII_path, task=task, metric='PCKh')

        self.__index = [0 for _ in range(len(self.__FLIC))]\
                       + [1 for _ in range(len(self.__MPII))]
        random.shuffle(self.__index)
        self.__seeker = 0

    def __delete__(self):
        pass

    def reset(self):
        self.__FLIC.reset()
        self.__MPII.reset()
        random.shuffle(self.__index)
        self.__seeker = 0

    def getBatch(self, size):
        batch_rgb = []
        batch_heat = []
        batch_pose = []
        batch_threshold = []
        batch_mask = []

        for _ in range(size):
            if self.__seeker == len(self):
                break

            if self.__index[self.__seeker] == 0:
                rgb, heat, pose, threshold, mask = self.__FLIC.getBatch(1)
            elif self.__index[self.__seeker] == 1:
                rgb, heat, pose, threshold, mask = self.__MPII.getBatch(1)
            else:
                raise IndexError()

            batch_rgb.append(rgb[0])
            batch_heat.append(heat[0])
            batch_pose.append(pose[0])
            batch_threshold.append(threshold[0])
            batch_mask.append(mask[0])

            self.__seeker += 1

        return np.stack(batch_rgb), \
               np.stack(batch_heat), \
               np.stack(batch_pose), \
               np.stack(batch_threshold), \
               np.stack(batch_mask)

    def __len__(self):
        return len(self.__index)