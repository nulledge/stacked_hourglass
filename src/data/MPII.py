import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]
interface_path = os.path.join(script_path, '..')
sys.path.append(interface_path)

from data import *
from functools import lru_cache

''' MPII data reader

Attribs:
    NUMBER_OF_DATA: The number of data in MPII annotated as training data.
        MPII does not provide any test data, This reader divides training data
        into train/eval set.
    TRAIN_RATIO: The number of training data is TRAIN_RATIO * NUMBER_OF_DATA.
    JOINT_TO_INDEX: Mapping from JOINT enum to valid MPII annotation index.
        Also can be used as mask.
'''       
class MPII(DataInterface):
    
    NUMBER_OF_DATA = 25994 + 2889
    TRAIN_RATIO = 0.9    
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
    def __init__(self, root, task, metric):
        self.__extract_path = {'image': os.path.join(root, 'images'),
                               'annotation': os.path.join(root, 'mpii_human_pose_v1_u12_2')}
        self.__metric = metric
        
        self.__loadAnnotation()
        
        # handle the task set as a file.
        if not self.__validTaskSet(root, task):
            self.__refreshTaskSet(root)
        self.__taskSet = open(os.path.join(root, task + '.txt'), 'r')
        
        
    def __delete__(self):
        self.__taskSet.close()
        

    ''' Load MPII annotation matlab file.
    
    The annotation file is loaded into self.__annotation.
    '''
    def __loadAnnotation(self):
        matlab_path = os.path.join(self.__extract_path['annotation'], 'mpii_human_pose_v1_u12_1.mat')
        self.__annotation = scipy.io.loadmat(matlab_path)['RELEASE']
        
        
    ''' Check if the task set file is valid.
    
    The task set file is ${root}/${task}.txt.
    
    Args:
        path: Path to the task set file.
        task: 'train' or 'eval'.
    
    Return:
        True if task set file exists.
    '''
    def __validTaskSet(self, path, task):
        return os.path.exists(os.path.join(path, task + '.txt'))
    
    
    ''' Refresh task sets.
    
    Arg:
        path: Path to the task set file.
    
    The task set files are ${root}/train.txt and ${root}/eval.txt.
    '''
    def __refreshTaskSet(self, path):
        img_train = self.__annotation['img_train'][0, 0][0, :]
        indices = []

        for img_idx in range(len(img_train)):
            if img_train[img_idx] == 0:
                continue
            assert img_train[img_idx] == 1

            humans_in_image = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'].shape[1]

            for r_idx in range(humans_in_image):
                try:
                    assert self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['objpos'][0, 0]['y'][0, 0]
                    indices.append((img_idx, r_idx))
                    
                except Exception as e:
                    print('Wrong annotated as train data.',
                          e,
                          self.__annotation['annolist'][0, 0][0, img_idx]['image'][0, 0]['name'][0])

        MPII.NUMBER_OF_DATA = len(indices)
        random.shuffle(indices)
        
        with open(os.path.join(path, 'train.txt'), 'w') as train_set:
            for i in range(int(MPII.TRAIN_RATIO * MPII.NUMBER_OF_DATA)):
                train_set.write(str(indices[i][0]) + " " + str(indices[i][1]) + '\n')
                
        with open(os.path.join(path, 'eval.txt'), 'w') as eval_set:
            for i in range(int(MPII.TRAIN_RATIO * MPII.NUMBER_OF_DATA), MPII.NUMBER_OF_DATA):
                eval_set.write(str(indices[i][0]) + " " + str(indices[i][1]) + '\n')


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
    def getBatch(self, size):
        
        batch_index = self.__loadBatchIndex(size)
        
        batch_rgb = np.ndarray(shape = (len(batch_index), 256, 256, 3), dtype = np.float32)
        for index in range(len(batch_index)):
            batch_rgb[index][:, :, :] = self.__getRGB(batch_index[index])
            
        batch_heat = np.ndarray(shape = (len(batch_index), 64, 64, len(JOINT)), dtype = np.float32)
        for index in range(len(batch_index)):
            batch_heat[index][:, :, :] = self.__getHeat(batch_index[index])
            
        batch_threshold = []
        for index in range(len(batch_index)):
            batch_threshold.append(self.__getThreshold(batch_index[index]))
            
        return batch_rgb, batch_heat, batch_threshold, MPII.__getMasking()
    
    
    ''' Set to read data from initial.
    '''
    def reset(self):
        self.__taskSet.seek(0)
    
    
    ''' Read indices from task index file.
    
    If one epoch is done, do not return anymore.
    
    Args:
        size: Batch size.
        
    Return:
        Batch indices list from the task set.
    '''
    def __loadBatchIndex(self, size):    
        batch_index = []
        for _ in range(size):
            index = self.__taskSet.readline()
            if index == '':
                break
            img_idx, r_idx = index.split(' ')
            batch_index.append((int(img_idx), int(r_idx)))
        return batch_index
    
    
    ''' Calculate padding.
    
    Arg:
        index: Data index.
        
    Returns:
        Image path, original center, bounding box length, padding size.
    '''
    @lru_cache(maxsize = 32)
    def __padImage(self, index):
        img_idx, r_idx = index
        
        image_name = self.__annotation['annolist'][0, 0][0, img_idx]['image'][0, 0]['name'][0]
        image_path = os.path.join(self.__extract_path['image'], image_name)
        image_res = {}
        image_res['vertical'], image_res['horizontal'] = Image.open(image_path).size
        
        center = {
            'vertical': int(self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['objpos'][0, 0]['y'][0, 0]),
            'horizontal': int(self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['objpos'][0, 0]['x'][0, 0])
        }
        length = int(self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['scale'][0, 0] * 200)
        
        pad = {}
        pad['vertical'] = {'up':0, 'down':0}
        pad['horizontal'] = {'left':0, 'right':0}
        
        if center['vertical'] - length//2 < 0:
            pad['vertical']['up'] = length//2 - center['vertical']
        if center['vertical'] + length//2 >= image_res['vertical']:
            pad['vertical']['down'] = center['vertical'] + length//2 - image_res['vertical']
        if center['horizontal'] - length//2 < 0:
            pad['horizontal']['left'] = length//2 - center['horizontal']
        if center['horizontal'] + length//2 >= image_res['horizontal']:
            pad['horizontal']['right'] = center['horizontal'] + length//2 - image_res['horizontal']
        
        return image_path, center, length, pad
    
    
    ''' Get RGB image.
    
    Arg:
        index: Data index.

    Return:
        RGB image of 256*256 px.
    '''
    def __getRGB(self, index):
        image_path, center, length, pad = self.__padImage(index)
        
        return scipy.misc.imresize(
            np.pad(
                imageio.imread(image_path),
                (
                    (pad['vertical']['up'], pad['vertical']['down']),
                    (pad['horizontal']['left'], pad['horizontal']['right']),
                    (0, 0)),
                'constant', constant_values = (0, 0)
            )[center['vertical'] + pad['vertical']['up'] - length//2
                  :center['vertical'] + pad['vertical']['down'] + length//2,
              center['horizontal'] + pad['horizontal']['left'] - length//2
                  :center['horizontal'] + pad['horizontal']['right'] + length//2,
              :],
            (256, 256)
        )
    
    
    ''' Get Heatmap images.
    
    Arg:
        index: Data index.
        
    Return:
        Heatmap images of 64*64 px for all joints.
    '''
    def __getHeat(self, index):
        heatmaps = np.ndarray(shape = (64, 64, len(JOINT)))
        _, center, length, pad = self.__padImage(index)
        img_idx, r_idx = index
        
        keypoints_ref = self.__annotation['annolist'][0, 0][0, img_idx]['annorect'][0, r_idx]['annopoints'][0, 0]['point']
        
        for joint in JOINT:
            heatmaps[:, :, joint.value] = 0
            if joint not in MPII.JOINT_TO_INDEX:
                continue
            n_joint = keypoints_ref.shape[1]
            for joint_idx in range(n_joint):
                try:
                    visible = not (not keypoints_ref[0, joint_idx]['is_visible']
                        or keypoints_ref[0, joint_idx]['is_visible'] == '0'
                        or keypoints_ref[0, joint_idx]['is_visible'] == 0)
                except:
                    visible = True
                tag = keypoints_ref[0, joint_idx]['id'][0, 0]

                if MPII.JOINT_TO_INDEX[joint] != tag or visible == False:
                    continue

                scale = 64 / length
                heatmaps[:, :, joint.value] = generateHeatmap(64, 1,
                      [(int(keypoints_ref[0, joint_idx]['y'][0, 0]) 
                          + length//2
                          - center['vertical']) * scale,
                      (int(keypoints_ref[0, joint_idx]['x'][0, 0])
                        + length//2
                        - center['horizontal']) * scale])
                    
        return heatmaps
    
    
    ''' Calculate PCKh threshold.
    
    Arg:
        index: Data index.
        
    Return:
        PCKh threshold
    '''
    def __getThresholdPCKh(self, index):
        _, _, length, _ = self.__padImage(index)
        img_idx, r_idx = index
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
    def __getThreshold(self, index):
        if self.__metric == 'PCK':
            return self.__getThresholdPCK(index)
        elif self.__metric == 'PCKh':
            return self.__getThresholdPCKh(index)
    
    
    ''' Get mask of MPII.
    
    Return:
        Binary list representing mask of MPII.
    ''' 
    def __getMasking():
        return [(lambda joint: joint in MPII.JOINT_TO_INDEX)(joint) for joint in JOINT]