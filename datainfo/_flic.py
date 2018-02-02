import os
import random
from functools import lru_cache

import numpy as np
import scipy
from tqdm import tqdm

from ._joint import JOINT
from .utils import cropRGB, transformImage, transformPosition, generateHeatmap

''' FLIC data reader

Attribs:
    NUMBER_OF_DATA: The number of data in FLIC.
    TRAIN_RATIO: The number of training data is TRAIN_RATIO * NUMBER_OF_DATA.
    JOINT_TO_INDEX: Mapping from JOINT enum to valid FLIC annotation index.
        Also can be used as mask.

'''


class FLIC:
    NUMBER_OF_DATA = 5003
    TRAIN_RATIO = 0.9
    ROTATE_DEGREE = 30
    SCALE_FACTOR = 0.25
    JOINT_TO_INDEX = {
        JOINT.L_Shoulder: 0,
        JOINT.L_Elbow: 1,
        JOINT.L_Wrist: 2,
        JOINT.R_Shoulder: 3,
        JOINT.R_Elbow: 4,
        JOINT.R_Wrist: 5,
        JOINT.L_Hip: 6,
        JOINT.R_Hip: 9
    }

    ''' Initialize FLIC reader.

    Args:
        root: The root path to FLIC data.
        task: 'train' or 'eval'.
        metric: 'PCK' or 'PCKt' only.
    '''

    def __init__(self, root, batch_size, task='train', shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.task = task
        self.shuffle = shuffle
        self.joints = len(JOINT)

        self.__extract_path = os.path.join(root, 'FLIC')  # default dataset folder
        self.__imageset_paths = {
            'train': os.path.join(self.__extract_path, "flic-train.txt"),
            'eval': os.path.join(self.__extract_path, "flic-eval.txt"),
        }

        # load annotation
        matlab_path = os.path.join(self.__extract_path, 'examples.mat')
        self.__annotation = scipy.io.loadmat(matlab_path)['examples']

        # handle the task set as a file.
        if not any(os.path.exists(path) for _, path in self.__imageset_paths.items()):
            print('refresh task set at: %s' % root)
            self.__refreshTaskSet()

        self.__seeker = 0
        self.__imageset = []  # tuple of image information. (index, rotation degree, scaling factor)
        with open(self.__imageset_paths[task], 'r') as handle:
            while True:
                image_info = handle.readline()
                if image_info == '':
                    break
                index, rotate, scale = image_info.split(' ')
                self.__imageset.append((int(index), float(rotate), float(scale)))

        if self.shuffle:
            random.shuffle(self.__imageset)

    ''' Refresh task sets.

    Arg:
        path: Path to the task set file.

    The task set files are ${root}/flic-train.txt and ${root}/flic-eval.txt.
    '''

    def __refreshTaskSet(self):
        def get_rand(pivot, factor):
            return random.uniform(pivot - factor, pivot + factor)

        indices = [index for index in range(FLIC.NUMBER_OF_DATA)]

        rotate = 0.0
        scale = 1.0
        with open(self.__imageset_paths['train'], 'w') as train_set:
            for idx in tqdm(indices[0:int(FLIC.TRAIN_RATIO * FLIC.NUMBER_OF_DATA)]):
                rand_rot, rand_scale = get_rand(0.0, FLIC.ROTATE_DEGREE), get_rand(1.0, FLIC.SCALE_FACTOR)
                # train_set.write("%d %f %f\n" % (idx, rotate, scale))
                train_set.write("%d %f %f\n" % (idx, rand_rot, rand_scale))  # image augmentation

        with open(self.__imageset_paths['eval'], 'w') as eval_set:
            for idx in tqdm(indices[int(FLIC.TRAIN_RATIO * FLIC.NUMBER_OF_DATA):]):
                eval_set.write("%d %f %f\n" % (idx, rotate, scale))  # indices, rotate, scale

    ''' Read data with the number of specific size.

    If one epoch is done, the batch size of return tuple could be less than the parameter.

    Args:
        size: Batch size.

    Returns:
        Tuple of RGB images, heatmaps, position, metric threshold and masks.
        RGB images are shape of (size, 256, 256, 3).
        Heatmaps are shape of (size, 64, 64, joint).
        Positions are shape of (size, joint, 2).
        Metric threshold is a single value.
        Masks are shape of (joint).
    '''

    def __getMiniBatch(self, imageset_batch):
        # batch_index = self.__loadBatchIndex(size)

        batch_rgb = np.ndarray(shape=(len(imageset_batch), 256, 256, 3), dtype=np.float32)
        batch_heatmap = np.ndarray(shape=(len(imageset_batch), 64, 64, self.joints), dtype=np.float32)
        batch_masking = np.ndarray(shape=(len(imageset_batch), self.joints), dtype=np.bool)
        batch_pose = np.ndarray(shape=(len(imageset_batch), self.joints, 2), dtype=np.float32)
        batch_threshold = []

        for idx, image_info in enumerate(imageset_batch):
            batch_rgb[idx][:, :, :] = self.__getRGB(image_info)
            batch_heatmap[idx][:, :, :], batch_masking[idx][:] = self.__getHeatAndMasking(image_info)
            batch_pose[idx][:, :] = self.__getPosition(image_info)
            batch_threshold.append(self.__getThreshold())

        return batch_rgb, batch_heatmap, batch_pose, batch_threshold, batch_masking

    ''' Set to read data from initial.
    '''

    def reset(self):
        self.__seeker = 0
        if self.shuffle:
            random.shuffle(self.__imageset)

    ''' Get the size of data.
    '''

    def __len__(self):
        return len(self.__imageset)

    def __iter__(self):
        return self

    def __next__(self):
        batch_imageset = self.__getImageSetBatch()
        if not len(batch_imageset):
            self.reset()
            raise StopIteration

        return self.__getMiniBatch(batch_imageset)

    ''' Read indices from task index file.

    If one epoch is done, do not return anymore.

    Args:
        size: Batch size.

    Return:
        Batch indices list from the task set.
    '''

    def __getImageSetBatch(self):
        prev_seeker = self.__seeker
        self.__seeker += self.batch_size
        return self.__imageset[prev_seeker:self.__seeker]

        # for _ in range(size):
        #     if self.__seeker == len(self.__imageset):
        #         break
        #     batch_index.append(self.__imageset[self.__seeker])
        #     self.__seeker += 1
        # return batch_index

    ''' Calculate padding.

    Arg:
        index: Data index.

    Returns:
        Center position, length of bounding box and padding to each side.
    '''

    @lru_cache(maxsize=32)
    def __getPadding(self, index):
        torso = {}
        torso['left'], torso['up'], torso['right'], torso['down'] = FLIC.__squeeze(self.__annotation,
                                                                                   ['torsobox', index])
        torso = dict([(key, int(value)) for key, value in torso.items()])

        resolution = {}
        resolution['vertical'], resolution['horizontal'], _ = FLIC.__squeeze(self.__annotation, ['imgdims', index])

        center = {
            'horizontal': (torso['right'] + torso['left']) // 2,
            'vertical': resolution['vertical'] // 2
        }

        pad = {
            'horizontal': {
                'left': 0, 'right': 0
            },
            'vertical': {
                'up': 0, 'down': 0
            }
        }

        length = resolution['vertical']

        if center['horizontal'] - length // 2 < 0:
            pad['horizontal']['left'] = length // 2 - center['horizontal']
        if center['horizontal'] + length // 2 >= resolution['horizontal']:
            pad['horizontal']['right'] = center['horizontal'] + length // 2 - resolution['horizontal']

        return center, length, pad, resolution

    ''' Get RGB image.

    Arg:
        index: Data index.

    Return:
        RGB image of 256*256 px.
    '''

    def __getRGB(self, image_info):
        index, rotate, scale = image_info
        center, length, pad, _ = self.__getPadding(index)
        image_path = os.path.join(
            self.__extract_path,
            'images',
            FLIC.__squeeze(self.__annotation, ['filepath', index]).item()
        )
        image = cropRGB(image_path, center, length, pad)
        image = transformImage(image, rotate, scale)

        return image

    ''' Get Heatmap images.

    Arg:
        index: Data index.

    Return:
        Heatmap images of 64*64 px for all joints.
    '''

    def __getHeatAndMasking(self, image_info):
        index, rotate, scale = image_info
        maskings = FLIC.__getMasking()
        heatmaps = np.ndarray(shape=(64, 64, self.joints), dtype=np.float32)
        center, length, _, resolution = self.__getPadding(index)

        horizontal, vertical = FLIC.__squeeze(self.__annotation, [index, 'coords'])

        for joint in JOINT:
            heatmaps[:, :, joint.value] = 0
            if joint not in FLIC.JOINT_TO_INDEX:
                continue
            resize = 64 / length
            outlier = int(vertical[FLIC.JOINT_TO_INDEX[joint]]) not in range(0, resolution['vertical']) \
                      or int(horizontal[FLIC.JOINT_TO_INDEX[joint]]) not in range(0, resolution['horizontal'])
            pose = {
                'vertical': vertical[FLIC.JOINT_TO_INDEX[joint]] * resize,
                'horizontal': (horizontal[FLIC.JOINT_TO_INDEX[joint]] - center['horizontal'] + length // 2) * resize
            }
            outlier = outlier \
                      or int(pose['vertical']) not in range(0, 64) \
                      or int(pose['horizontal']) not in range(0, 64)
            pose, outlier_after_transform = transformPosition(
                pose,
                rotate=rotate,
                scale=scale
            )
            outlier = outlier or outlier_after_transform
            if outlier:
                maskings[joint.value] = False
                continue
            heatmaps[:, :, joint.value] = generateHeatmap(64, 1, [pose['vertical'], pose['horizontal']])

        return heatmaps, maskings

    ''' Get joint positions.

    Arg:
        index: Data index.

    Return:
        Positions for all joints.
    '''

    def __getPosition(self, image_info):
        index, rotate, scale = image_info
        position = np.ndarray(shape=(self.joints, 2))
        center, length, pad, resolution = self.__getPadding(index)

        horizontal, vertical = FLIC.__squeeze(self.__annotation, [index, 'coords'])

        for joint in JOINT:
            if joint not in FLIC.JOINT_TO_INDEX:
                continue
            else:
                resize = 64 / length
                pose = {
                    'vertical': vertical[FLIC.JOINT_TO_INDEX[joint]] * resize,
                    'horizontal': (horizontal[FLIC.JOINT_TO_INDEX[joint]] - center['horizontal'] + length // 2) * resize
                }
                pose, _ = transformPosition(
                    pose,
                    rotate=rotate,
                    scale=scale
                )

                position[joint.value, :] = [pose['vertical'], pose['horizontal']]

        return position

    ''' Calculate PCK threshold.

    Arg:
        index: Data index.

    Return:
        PCK threshold
    '''

    def __getThresholdPCK(self):
        return 64

    ''' Get threshold of metric.

    Arg:
        index: Data index.

    Return:
        Threshold of metric.
    '''

    def __getThreshold(self):
        return self.__getThresholdPCK()
        # if self.__metric == 'PCK':
        #     return self.__getThresholdPCK(index)
        # elif self.__metric == 'PCKh':
        #     return self.__getThresholdPCKh(index)

    ''' Get mask of FLIC.

    Return:
        Binary list representing mask of FLIC.
    '''

    @staticmethod
    def __getMasking():
        return [(lambda joint: joint in FLIC.JOINT_TO_INDEX)(joint) for joint in JOINT]

    ''' FLIC annotation parsing utility.

    To parse FLIC annotation, we have to travle with unnecessary index 0, like annotation['label'][0][0].
    This utility remove these unnecessary indices.

    Args:
        annotation: Numpy structured ndarray representing annotation.
        indices: Meaningful indices to parse.

    Return:
        Parsed FLIC annotation.
    '''

    @staticmethod
    def __squeeze(annotation, indices):
        if len(indices) == 0:
            return np.squeeze(annotation)
        return FLIC.__squeeze(np.squeeze(annotation)[indices[0]], indices[1:])
