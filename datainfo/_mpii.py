import math
import os
import random
from functools import lru_cache

import imageio
import numpy as np
import scipy
import skimage.transform, skimage.io
from tqdm import tqdm

from ._joint import JOINT
from .utils import generateHeatmap

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
        def rand(x):
            return max(-2 * x, min(2 * x, random.gauss(0, 1) * x))

        batch_rgb = np.ndarray(shape=(len(imageset_batch), 256, 256, 3), dtype=np.float32)
        batch_heatmap = np.zeros(shape=(len(imageset_batch), 64, 64, self.joints), dtype=np.float32)
        batch_keypoint = np.zeros(shape=(len(imageset_batch), self.joints, 2), dtype=np.float32)
        batch_masking = np.zeros(shape=(len(imageset_batch), self.joints), dtype=np.bool)
        batch_threshold = np.zeros(shape=(len(imageset_batch)), dtype=np.float32)

        for idx, image_info in enumerate(imageset_batch):
            img_idx, r_idx, _, _ = image_info
            annotation = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]

            # scale augmentation
            scale = annotation['scale'][0, 0] * 1.25
            if self.task == 'train':
                scale *= 2 ** rand(MPII.SCALE_FACTOR)

            # rotation augmentation
            rotate = 0.0
            if random.random() <= 0.4 and self.task == 'train':
                rotate = rand(MPII.ROTATE_DEGREE)  # rotate

            batch_rgb[idx], batch_heatmap[idx], batch_keypoint[idx], batch_masking[idx], \
            batch_threshold[idx] = self.__getRGB(image_info, scale, rotate)

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

    def __getRGB(self, image_info, scale, rotate):
        img_idx, r_idx, _, _ = image_info
        path = self.__getImagePath(image_info)
        annotation = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]


        image = skimage.io.imread(path)
        image = skimage.img_as_float(image)

        POSITION = annotation['objpos'][0, 0]
        OUTPUT_RES = 256
        LENGTH = int(200 * scale)
        HALF_LEN = LENGTH // 2

        x_center = POSITION['x'][0, 0]
        y_center = POSITION['y'][0, 0] + 15.0 * scale
        X_CENTER = POSITION['x'][0, 0]
        Y_CENTER = POSITION['y'][0, 0] + 15.0 * scale

        tmp_image = image
        height, width, _ = image.shape
        HEIGHT, WIDTH, _ = image.shape

        crop_ratio = (200 * scale) / OUTPUT_RES
        if crop_ratio < 2:
            crop_ratio = 1
        else:
            new_height = math.floor(height / crop_ratio)
            new_width = math.floor(width / crop_ratio)

            if max([new_height, new_width]) < 2:
                raise ValueError("?????????????????????????????")
            else:
                tmp_image = skimage.transform.resize(image, (new_height, new_width))
                height, width = new_height, new_width

        x_center /= crop_ratio
        y_center /= crop_ratio
        scale /= crop_ratio

        x_ul = int(x_center - 200 * scale / 2)
        y_ul = int(y_center - 200 * scale / 2)

        x_br = int(x_center + 200 * scale / 2)
        y_br = int(y_center + 200 * scale / 2)

        if crop_ratio >= 2:  # force image size 256 x 256
            x_br -= x_br - x_ul - OUTPUT_RES
            y_br -= y_br - y_ul - OUTPUT_RES

        pad_length = math.ceil((math.sqrt((x_ul - x_br) ** 2 + (y_ul - y_br) ** 2) - (x_br - x_ul)) / 2)

        if rotate != 0:
            x_ul -= pad_length
            y_ul -= pad_length
            x_br += pad_length
            y_br += pad_length

        x_ul = int(x_ul)
        y_ul = int(y_ul)
        x_br = int(x_br)
        y_br = int(y_br)
        width = int(width)
        height = int(height)
        pad_length = int(pad_length)

        src = [max(0, y_ul), min(height, y_br), max(0, x_ul), min(width, x_br)]
        dst = [max(0, -y_ul), min(height, y_br) - y_ul, max(0, -x_ul), min(width, x_br) - x_ul]

        new_image = np.zeros([y_br - y_ul, x_br - x_ul, 3], dtype=np.float32)
        new_image[dst[0]:dst[1], dst[2]:dst[3], :] = tmp_image[src[0]:src[1], src[2]:src[3], :]


        if rotate != 0:
            new_image = skimage.transform.rotate(new_image, rotate)
            new_height, new_width, _ = new_image.shape
            new_image = new_image[
                        pad_length:new_height - pad_length,
                        pad_length:new_width - pad_length,
                        :]

        if crop_ratio < 2:
            new_image = skimage.transform.resize(new_image, (OUTPUT_RES, OUTPUT_RES))

        gt_image = new_image
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

        for idx in range(keypoints.shape[1]):  # num of joint
            joint_id = keypoints[0, idx]['id'][0, 0]

            if joint_id not in MPII.ID_TO_JOINT:
                raise ValueError('Wrong joint INDEX! (%d)' % joint_id)

            resize = 64 / LENGTH
            gt_x, gt_y = (int(keypoints[0, idx]['x'][0, 0]), int(keypoints[0, idx]['y'][0, 0]))
            outlier = (gt_x < 0 or gt_x >= WIDTH) or (gt_y < 0 or gt_y >= HEIGHT)

            gt_x = (gt_x - (X_CENTER - HALF_LEN)) * resize  # space change: original image >> crop image
            gt_y = (gt_y - (Y_CENTER - HALF_LEN)) * resize

            x_crop = gt_x - 64 // 2  # space change: crop image >> crop center
            y_crop = gt_y - 64 // 2

            cos = math.cos(rotate * math.pi / 180)
            sin = math.sin(rotate * math.pi / 180)
            y_crop_rot = cos * y_crop - sin * x_crop
            x_crop_rot = sin * y_crop + cos * x_crop

            gt_x_rot = x_crop_rot + 64 // 2
            gt_y_rot = y_crop_rot + 64 // 2

            outlier = outlier or (gt_x_rot < 0 or gt_x_rot >= 64) or (gt_y_rot < 0 or gt_y_rot >= 64)

            if outlier:
                continue

            gt_heatmaps[:, :, MPII.ID_TO_JOINT[joint_id].value] = generateHeatmap(64, gt_y_rot, gt_x_rot)
            gt_keypoints[MPII.ID_TO_JOINT[joint_id].value, :] = [gt_y_rot, gt_x_rot]
            gt_maskings[MPII.ID_TO_JOINT[joint_id].value] = True

        gt_threshold = np.linalg.norm(
            np.array([int(annotation['y1'][0, 0]), int(annotation['x1'][0, 0])])
            - np.array([int(annotation['y2'][0, 0]), int(annotation['x2'][0, 0])])) * 64 / LENGTH
        return gt_image, gt_heatmaps, gt_keypoints, gt_maskings, gt_threshold


    ''' Get mask of MPII.
    
    Return:
        Binary list representing mask of MPII.
    '''


    @staticmethod
    def __getMasking():
        maskings = [False] * len(JOINT)
        return maskings
