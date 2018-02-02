import random

import numpy as np

from ._flic import FLIC
from ._mpii import MPII


class MIX:
    def __init__(self, root, batch_size, task='train', shuffle=True):
        assert task == 'train'

        self.root = root
        self.batch_size = batch_size
        self.task = task
        self.shuffle = shuffle

        self.__FLIC = FLIC(root=root, batch_size=1, task=task)
        self.__MPII = MPII(root=root, batch_size=1, task=task)

        self.__order = [0 for _ in range(len(self.__FLIC))] \
                       + [1 for _ in range(len(self.__MPII))]
        if self.shuffle:
            random.shuffle(self.__order)
        self.__seeker = 0

    def __delete__(self):
        pass

    def reset(self):
        if self.shuffle:
            random.shuffle(self.__order)
        self.__seeker = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__seeker >= len(self):
            self.reset()
            self.__FLIC.reset()
            self.__MPII.reset()
            raise StopIteration

        return self.getBatch()

    def getBatch(self):
        batch_rgb = []
        batch_heat = []
        batch_pose = []
        batch_threshold = []
        batch_mask = []

        prev_seeker = self.__seeker
        self.__seeker += self.batch_size
        mini_batch = self.__order[prev_seeker:self.__seeker]

        for target_set in mini_batch:

            try:
                if target_set == 0:
                    rgb, heat, pose, threshold, mask = self.__FLIC.__next__()
                elif target_set == 1:
                    rgb, heat, pose, threshold, mask = self.__MPII.__next__()
                else:
                    raise IndexError()
            except Exception as e:
                print(e)

            batch_rgb.append(rgb[0])
            batch_heat.append(heat[0])
            batch_pose.append(pose[0])
            batch_threshold.append(threshold[0])
            batch_mask.append(mask[0])

        return np.stack(batch_rgb), np.stack(batch_heat), \
               np.stack(batch_pose), np.stack(batch_threshold), np.stack(batch_mask)

    def __len__(self):
        return len(self.__order)