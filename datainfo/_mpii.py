import os
import random
from functools import lru_cache

import numpy as np
import scipy
from PIL import Image
from tqdm import tqdm

from ._joint import JOINT
from .utils import cropRGB, transformImage, transformPosition, generateHeatmap

''' MPII data reader

Attrib:
    NUMBER_OF_DATA: The number of data in MPII annotated as training data.
        MPII does not provide any test data, This reader divides training data
        into train/eval set.
    TRAIN_RATIO: The number of training data is TRAIN_RATIO * NUMBER_OF_DATA.
    JOINT_TO_INDEX: Mapping from JOINT enum to valid MPII annotation index.
        Also can be used as mask.
'''


class MPII:
    NUMBER_OF_DATA = 25994 + 2889
    TRAIN_RATIO = 0.9
    ROTATE_DEGREE = 30
    SCALE_FACTOR = 0.25
    JOINT_TO_INDEX = {
        JOINT.R_Ankle: 0,
        JOINT.R_Knee: 1,
        JOINT.R_Hip: 2,
        JOINT.L_Hip: 3,
        JOINT.L_Knee: 4,
        JOINT.L_Ankle: 5,
        JOINT.M_Pelvis: 6,
        JOINT.M_Thorax: 7,
        JOINT.M_UpperNeck: 8,
        JOINT.M_HeadTop: 9,
        JOINT.R_Wrist: 10,
        JOINT.R_Elbow: 11,
        JOINT.R_Shoulder: 12,
        JOINT.L_Shoulder: 13,
        JOINT.L_Elbow: 14,
        JOINT.L_Wrist: 15
    }

    ''' Initialize MPII reader.

    Args:
        root: The root path to MPII data.
        task: 'train' or 'eval'.
        metric: 'PCKh' only.
    '''

    def __init__(self, root, batch_size, task='train', shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.task = task
        self.shuffle = shuffle
        self.joints = len(JOINT)

        self.__extract_path = os.path.join(root, 'MPII')
        self.__annotation_path = os.path.join(self.__extract_path, 'mpii_human_pose_v1_u12_2')
        self.__image_path = os.path.join(self.__extract_path, 'images')
        self.__imageset_paths = {
            'train': os.path.join(self.__extract_path, "mpii-train.txt"),
            'eval': os.path.join(self.__extract_path, "mpii-eval.txt"),
        }

        # load annotation
        matlab_path = os.path.join(self.__annotation_path, 'mpii_human_pose_v1_u12_1.mat')
        self.__annotation = scipy.io.loadmat(matlab_path)['RELEASE']

        # handle the task set as a file.
        if not any(os.path.exists(path) for _, path in self.__imageset_paths.items()):
            print('refresh task set at: %s' % root)
            self.__refreshTaskSet()

        # if not self.__validTaskSet(root, task):
        #     self.__refreshTaskSet(root)
        # self.__taskSet = open(os.path.join(root, task + '.txt'), 'r')

        self.__seeker = 0
        self.__imageset = []
        with open(self.__imageset_paths[task], 'r') as handle:
            while True:
                index = handle.readline()
                if index == '':
                    break
                img_idx, r_idx, rotate, scale = index.split(' ')
                self.__imageset.append((int(img_idx), int(r_idx), float(rotate), float(scale)))

        if self.shuffle:
            random.shuffle(self.__imageset)

    ''' Refresh task sets.

    Arg:
        path: Path to the task set file.

    The task set files are ${root}/train.txt and ${root}/eval.txt.
    '''

    def __refreshTaskSet(self):
        img_train = self.__annotation['img_train'][0, 0][0, :]
        indices = []

        for img_idx in range(len(img_train)):
            if img_train[img_idx] == 0:
                continue
            assert img_train[img_idx] == 1

            humans_in_image = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'].shape[1]

            for r_idx in range(humans_in_image):
                try:
                    assert self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['objpos'][0, 0]['y'][
                        0, 0]
                    indices.append((img_idx, r_idx))

                except Exception as e:
                    print('Wrong annotated as train data.',
                          e,
                          self.__annotation['annolist'][0, 0][0, img_idx]['image'][0, 0]['name'][0])

        MPII.NUMBER_OF_DATA = len(indices)  # update length because wrong annotation

        def get_rand(pivot, factor):
            return random.uniform(pivot - factor, pivot + factor)

        rotate = 0.0
        scale = 1.0
        with open(self.__imageset_paths['train'], 'w') as train_set:
            for idx in tqdm(indices[0:int(MPII.TRAIN_RATIO * MPII.NUMBER_OF_DATA)]):
                rand_rot, rand_scale = get_rand(0.0, MPII.ROTATE_DEGREE), get_rand(1.0, MPII.SCALE_FACTOR)
                train_set.write("%d %d %f %f\n" % (idx[0], idx[1], rotate, scale))
                train_set.write("%d %d %f %f\n" % (idx[0], idx[1], rand_rot, rand_scale))  # image augmentation

        with open(self.__imageset_paths['eval'], 'w') as eval_set:
            for idx in tqdm(indices[int(MPII.TRAIN_RATIO * MPII.NUMBER_OF_DATA):]):
                eval_set.write("%d %d %f %f\n" % (idx[0], idx[1], rotate, scale))

    ''' Read data with the number of specific size.

    If one epoch is done, the batch size of return tuple could be less than the parameter.

    Args:
        size: Batch size.

    Returns:
        Tuple of RGB images, heatmaps, metric threshold and masks.
        RGB images are shape of (size, 256, 256, 3).
        Heatmaps are shape of (size, 64, 64, joint).
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
            batch_threshold.append(self.__getThreshold(image_info))

        return batch_rgb, batch_heatmap, batch_pose, batch_threshold, batch_masking

    ''' Set to read data from initial.
    '''

    def reset(self):
        self.__seeker = 0
        if self.shuffle:
            random.shuffle(self.__imageset)

    ''' Get data size.
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

    # def __loadBatchIndex(self, size):
    #     batch_index = []
    #     for _ in range(size):
    #         if self.__seeker == len(self.__index):
    #             break
    #         batch_index.append(self.__index[self.__seeker])
    #         self.__seeker += 1
    #     return batch_index

    ''' Get image path.

    Arg:
        index: Data index.

    Return:
        Path to image.
    '''

    @lru_cache(maxsize=32)
    def __getImagePath(self, image_info):
        img_idx, r_idx, _, _ = image_info

        image_name = self.__annotation['annolist'][0, 0][0, img_idx]['image'][0, 0]['name'][0]
        return os.path.join(self.__image_path, image_name)

    ''' Calculate padding.

    Arg:
        index: Data index.

    Returns:
        Image path, original center, bounding box length, padding size.
    '''

    @lru_cache(maxsize=32)
    def __getPadding(self, image_info):
        img_idx, r_idx, _, _ = image_info

        path = self.__getImagePath(image_info)
        resolution = dict()
        resolution['horizontal'], resolution['vertical'] = Image.open(path).size

        center = {
            'vertical': int(
                self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['objpos'][0, 0]['y'][0, 0]),
            'horizontal': int(
                self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['objpos'][0, 0]['x'][0, 0])
        }
        length = int(self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['scale'][0, 0] * 200)

        pad = {
            'horizontal': {
                'left': 0, 'right': 0
            },
            'vertical': {
                'up': 0, 'down': 0
            }
        }

        if center['vertical'] - length // 2 < 0:
            pad['vertical']['up'] = length // 2 - center['vertical']
        if center['vertical'] + length // 2 >= resolution['vertical']:
            pad['vertical']['down'] = center['vertical'] + length // 2 - resolution['vertical']
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
        center, length, pad, _ = self.__getPadding(image_info)
        path = self.__getImagePath(image_info)
        _, _, rotate, scale = image_info
        image = cropRGB(path, center, length, pad)
        image = transformImage(image, rotate, scale)
        return image

    ''' Get Heatmap images.

    Arg:
        index: Data index.

    Return:
        Heatmap images of 64*64 px for all joints.
    '''

    def __getHeatAndMasking(self, image_info):
        img_idx, r_idx, rotate, scale = image_info
        maskings = MPII.__getMasking()
        heatmaps = np.ndarray(shape=(64, 64, self.joints), dtype=np.float32)
        center, length, pad, resolution = self.__getPadding(image_info)

        keypoints_ref = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['annopoints'][0, 0][
            'point']

        for joint in JOINT:
            heatmaps[:, :, joint.value] = 0
            if joint not in MPII.JOINT_TO_INDEX:
                continue
            n_joint = keypoints_ref.shape[1]
            for joint_idx in range(n_joint):
                tag = keypoints_ref[0, joint_idx]['id'][0, 0]

                if MPII.JOINT_TO_INDEX[joint] != tag:
                    continue

                resize = 64 / length
                outlier = int(keypoints_ref[0, joint_idx]['y'][0, 0]) not in range(0, resolution['vertical']) \
                          or int(keypoints_ref[0, joint_idx]['x'][0, 0]) not in range(0, resolution['horizontal'])
                pose = {
                    'vertical': (keypoints_ref[0, joint_idx]['y'][0, 0] + length // 2 - center['vertical']) * resize,
                    'horizontal': (keypoints_ref[0, joint_idx]['x'][0, 0] + length // 2 - center['horizontal']) * resize
                }
                outlier = outlier \
                          or int(pose['vertical']) not in range(0, 64) \
                          or int(pose['horizontal']) not in range(0, 64)
                pose, outlier_after_transform = transformPosition(pose, rotate, scale)
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
        img_idx, r_idx, rotate, scale = image_info
        positions = np.ndarray(shape=(self.joints, 2))
        center, length, pad, _ = self.__getPadding(image_info)

        keypoints_ref = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['annopoints'][0, 0][
            'point']

        for joint in JOINT:
            if joint not in MPII.JOINT_TO_INDEX:
                continue
            n_joint = keypoints_ref.shape[1]
            for joint_idx in range(n_joint):
                tag = keypoints_ref[0, joint_idx]['id'][0, 0]

                if MPII.JOINT_TO_INDEX[joint] != tag:
                    continue

                resize = 64 / length
                pose = {
                    'vertical': (keypoints_ref[0, joint_idx]['y'][0, 0] + length // 2 - center['vertical']) * resize,
                    'horizontal': (keypoints_ref[0, joint_idx]['x'][0, 0] + length // 2 - center['horizontal']) * resize
                }
                pose, _ = transformPosition(pose, rotate, scale)

                positions[joint.value, :] = [pose['vertical'], pose['horizontal']]

        return positions

    ''' Calculate PCKh threshold.

    Arg:
        index: Data index.

    Return:
        PCKh threshold
    '''

    def __getThresholdPCKh(self, image_info):
        _, length, _, _ = self.__getPadding(image_info)
        img_idx, r_idx, _, _ = image_info
        human_ref = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]

        return np.linalg.norm(
            np.array([int(human_ref['y1'][0, 0]), int(human_ref['x1'][0, 0])])
            - np.array([int(human_ref['y2'][0, 0]), int(human_ref['x2'][0, 0])])) * 64 / length

    ''' Get threshold of metric.

    Arg:
        index: Data index.

    Return:
        Threshold of metric.
    '''

    def __getThreshold(self, image_info):
        return self.__getThresholdPCKh(image_info)

    # if self.__metric == 'PCK':
    #     return self.__getThresholdPCK(index)
    # elif self.__metric == 'PCKh':

    ''' Get mask of MPII.

    Return:
        Binary list representing mask of MPII.
    '''

    @staticmethod
    def __getMasking():
        return [(lambda joint: joint in MPII.JOINT_TO_INDEX)(joint) for joint in JOINT]
