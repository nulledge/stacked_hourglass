import math
import os
import random
from functools import lru_cache

import numpy as np
import scipy
import skimage.io
import skimage.transform
from tqdm import tqdm
from vectormath import Vector2

from ._joint import JOINT
from .utils import generateHeatmap, cropImage

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
    ID_TO_JOINT = {
        0: JOINT.R_Ankle,
        1: JOINT.R_Knee,
        2: JOINT.R_Hip,
        3: JOINT.L_Hip,
        4: JOINT.L_Knee,
        5: JOINT.L_Ankle,
        6: JOINT.M_Pelvis,
        7: JOINT.M_Thorax,
        8: JOINT.M_UpperNeck,
        9: JOINT.M_HeadTop,
        10: JOINT.R_Wrist,
        11: JOINT.R_Elbow,
        12: JOINT.R_Shoulder,
        13: JOINT.L_Shoulder,
        14: JOINT.L_Elbow,
        15: JOINT.L_Wrist
    }

    ''' Initialize MPII reader.

    Args:
        root: The root path to MPII data.
        task: 'train' or 'eval'.
        metric: 'PCKh' only.
    '''

    def __init__(self, root, batch_size, task='train', shuffle=True, augmentation=True):
        self.root = root
        self.batch_size = batch_size
        self.task = task
        self.shuffle = shuffle
        self.augmentation = augmentation
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

        rotate = 0.0
        scale = 1.0
        with open(self.__imageset_paths['train'], 'w') as train_set:
            for idx in tqdm(indices[0:int(MPII.TRAIN_RATIO * MPII.NUMBER_OF_DATA)]):
                train_set.write("%d %d %f %f\n" % (idx[0], idx[1], rotate, scale))
                # rand_rot, rand_scale = get_rand(0.0, MPII.ROTATE_DEGREE), get_rand(1.0, MPII.SCALE_FACTOR)
                # train_set.write("%d %d %f %f\n" % (idx[0], idx[1], rand_rot, rand_scale))  # image augmentation

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
        batch_rgb = np.ndarray(shape=(len(imageset_batch), 256, 256, 3), dtype=np.float32)
        batch_heatmap = np.zeros(shape=(len(imageset_batch), 64, 64, self.joints), dtype=np.float32)
        batch_keypoint = np.zeros(shape=(len(imageset_batch), self.joints, 2), dtype=np.float32)
        batch_masking = np.zeros(shape=(len(imageset_batch), self.joints), dtype=np.bool)
        batch_threshold = np.zeros(shape=(len(imageset_batch)), dtype=np.float32)

        for idx, image_info in enumerate(imageset_batch):
            batch_rgb[idx], batch_heatmap[idx], batch_keypoint[idx], batch_masking[idx], \
            batch_threshold[idx] = self.__getData(image_info)

        return batch_rgb, batch_heatmap, batch_keypoint, batch_threshold, batch_masking

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

    ''' Get RGB image.

    Arg:
        index: Data index.

    Return:
        RGB image of 256*256 px.
    '''

    def __getData(self, image_info):
        # gaussian random function
        def rand(x):
            return max(-2 * x, min(2 * x, random.gauss(0, 1) * x))

        img_idx, r_idx, _, _ = image_info
        path = self.__getImagePath(image_info)
        annotation = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]

        # scale and rotation augmentation
        rotate = 0.0
        scale = annotation['scale'][0, 0]
        if self.task == 'train':
            scale *= 1.25
            if self.augmentation:
                scale *= 2 ** rand(MPII.SCALE_FACTOR)
                rotate = rand(MPII.ROTATE_DEGREE) if random.random() <= 0.4 else 0

        # load image
        image = skimage.img_as_float(skimage.io.imread(path))

        # parse data annotation
        pos = annotation['objpos'][0, 0]
        # Small adjustment so cropping is less likely to take feet out
        center = Vector2(pos['x'][0, 0], pos['y'][0, 0] + 15.0 * scale)
        center.setflags(write=False)  # make immuatble

        gt_image = cropImage(image, center, scale, rotate, resolution=256)
        if self.task and self.augmentation:
            gt_image[:, :, 0] *= random.uniform(0.6, 1.4)
            gt_image[:, :, 1] *= random.uniform(0.6, 1.4)
            gt_image[:, :, 2] *= random.uniform(0.6, 1.4)
            gt_image = np.clip(gt_image, 0, 1)

        ###################################################

        assert gt_image.shape == (256, 256, 3)

        # calculate ground truth heatmaps and positions
        gt_maskings = MPII.__getMasking()
        gt_heatmaps = np.zeros(shape=(64, 64, self.joints), dtype=np.float32)
        gt_keypoints = np.zeros(shape=(self.joints, 2))

        keypoints = annotation['annopoints'][0, 0]['point']
        box_size = 200 * scale
        resize_ratio = box_size / 64

        for idx in range(keypoints.shape[1]):  # num of joint
            joint_id = keypoints[0, idx]['id'][0, 0]

            if joint_id not in MPII.ID_TO_JOINT:
                raise ValueError('Wrong joint INDEX! (%d)' % joint_id)

            keypoint = Vector2(keypoints[0, idx]['x'][0, 0], keypoints[0, idx]['y'][0, 0])
            keypoint -= (center - box_size / 2)  # space change: original image >> crop image
            keypoint /= resize_ratio

            # rotate image with center pivot
            if rotate != 0:
                keypoint -= 64 / 2  # space change: crop image >> crop center
                cos = math.cos(rotate * math.pi / 180)
                sin = math.sin(rotate * math.pi / 180)
                keypoint = Vector2(sin * keypoint.y + cos * keypoint.x, cos * keypoint.y - sin * keypoint.x)
                keypoint += 64 / 2

            outlier = min(keypoint) < 0 or max(keypoint) >= 64

            if outlier:
                continue

            gt_heatmaps[:, :, MPII.ID_TO_JOINT[joint_id].value] = generateHeatmap(64, keypoint.y, keypoint.x)
            gt_keypoints[MPII.ID_TO_JOINT[joint_id].value, :] = [keypoint.y, keypoint.x]
            gt_maskings[MPII.ID_TO_JOINT[joint_id].value] = True

        gt_threshold = np.linalg.norm(
            np.array([int(annotation['y1'][0, 0]), int(annotation['x1'][0, 0])])
            - np.array([int(annotation['y2'][0, 0]), int(annotation['x2'][0, 0])])) / resize_ratio
        return gt_image, gt_heatmaps, gt_keypoints, gt_maskings, gt_threshold

    ''' Get mask of MPII.
    
    Return:
        Binary list representing mask of MPII.
    '''

    @staticmethod
    def __getMasking():
        maskings = [False] * len(JOINT)
        return maskings
