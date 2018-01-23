import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]
interface_path = os.path.join(script_path, '..')
sys.path.append(interface_path)

from data import *
from functools import lru_cache

''' FLIC data reader

Attribs:
    NUMBER_OF_DATA: The number of data in FLIC.
    TRAIN_RATIO: The number of training data is TRAIN_RATIO * NUMBER_OF_DATA.
    JOINT_TO_INDEX: Mapping from JOINT enum to valid FLIC annotation index.
        Also can be used as mask.
    
'''       
class FLIC(DataInterface):
    
    NUMBER_OF_DATA = 5003
    TRAIN_RATIO = 0.9
    JOINT_TO_INDEX = {
        JOINT.L_Shoulder: 0,
        JOINT.L_Elbow: 1,
        JOINT.L_Wrist: 2,
        JOINT.R_Shoulder: 3,
        JOINT.R_Elbow: 4,
        JOINT.R_Wrist: 5,
        JOINT.L_Hip: 6,
        JOINT.R_Hip: 9,
        JOINT.L_Eye: 12,
        JOINT.R_Eye: 13,
        JOINT.M_Nose: 16
    }
    
    
    ''' Initialize FLIC reader.
    
    Args:
        root: The root path to FLIC data.
        task: 'train' or 'eval'.
        metric: 'PCK' or 'PCKt' only.
    '''
    def __init__(self, root, task, metric):
        self.__extract_path = os.path.join(root, 'FLIC')
        self.__metric = metric

        self.__loadAnnotation()
        
        # handle the task set as a file.
        if not self.__validTaskSet(root, task):
            self.__refreshTaskSet(root)
        self.__taskSet = open(os.path.join(root, task + '.txt'), 'r')

        
    def __delete__(self):
        self.__taskSet.close()

        
    ''' Load FLIC annotation matlab file.
    
    The annotation file is loaded into self.__annotation.
    '''
    def __loadAnnotation(self):
        matlab_path = os.path.join(self.__extract_path, 'examples.mat')
        self.__annotation = scipy.io.loadmat(matlab_path)['examples']
    
    
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
        indices = [i for i in range(FLIC.NUMBER_OF_DATA)]
        random.shuffle(indices)

        with open(os.path.join(path, 'train.txt'), 'w') as train_set:
            for i in range(int(FLIC.TRAIN_RATIO * FLIC.NUMBER_OF_DATA)):
                train_set.write(str(indices[i]) + '\n')

        with open(os.path.join(path, 'eval.txt'), 'w') as eval_set:
            for i in range(int(FLIC.TRAIN_RATIO * FLIC.NUMBER_OF_DATA), FLIC.NUMBER_OF_DATA):
                eval_set.write(str(indices[i]) + '\n')
    
    
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
            
        return batch_rgb, batch_heat, batch_threshold, FLIC.__getMasking()
    
    
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
            batch_index.append(int(index))
        return batch_index
    
    
    ''' Calculate padding.
    
    Arg:
        index: Data index.
        
    Returns:
        Padded center, original resolution, original torso position, padding size.
    '''
    @lru_cache(maxsize = 32)
    def __padImage(self, index):
        torso = {}
        torso['left'], torso['up'], torso['right'], torso['down'] = FLIC.__squeeze(self.__annotation, ['torsobox', index])
        torso['left'] = int(torso['left'])
        torso['up'] = int(torso['up'])
        torso['right'] = int(torso['right'])
        torso['down'] = int(torso['down'])
        
        center = {}
        center['horizontal'] = (torso['right'] + torso['left']) // 2
        
        resolution = {}
        resolution['vertical'], resolution['horizontal'], _ = FLIC.__squeeze(self.__annotation, ['imgdims', index])
        
        pad = {'left': 0, 'right': 0}
        if center['horizontal'] + resolution['vertical']//2 > resolution['horizontal']:
            pad['right'] = center['horizontal'] + resolution['vertical']//2 - resolution['horizontal']
        if center['horizontal'] - resolution['vertical']//2 < 0:
            pad['left'] = resolution['vertical']//2 - center['horizontal']
            center['horizontal'] += pad['left']
        
        return center, resolution, torso, pad
        
        
    ''' Get RGB image.
    
    Arg:
        index: Data index.

    Return:
        RGB image of 256*256 px.
    '''
    def __getRGB(self, index):
        center, resolution, _, pad = self.__padImage(index)
        return scipy.misc.imresize(
            np.pad(
                imageio.imread(
                    os.path.join(
                        self.__extract_path,
                        'images',
                        FLIC.__squeeze(self.__annotation, ['filepath', index]).item()
                    )
                ),
                (
                    (0, 0),
                    (pad['left'], pad['right']),
                    (0, 0)
                ),
                'constant', constant_values = (0, 0)
            )[:, center['horizontal'] - resolution['vertical']//2:center['horizontal'] + resolution['vertical']//2, :],
            (256, 256)
        )
    
    
    ''' Get Heatmap images.
    
    Arg:
        index: Data index.
        
    Return:
        Heatmap images of 64*64 px for all joints.
    '''
    def __getHeat(self, index):
        heatmaps = np.ndarray(shape = (64, 64, len(JOINT)), dtype = np.float32)
        center, resolution, _, pad = self.__padImage(index)
        
        horizontal, vertical = FLIC.__squeeze(self.__annotation, [index, 'coords'])
        
        for joint in JOINT:
            if joint not in FLIC.JOINT_TO_INDEX:
                heatmaps[:, :, joint.value] = 0
            else:
                scale = 64 / resolution['vertical']
                heatmaps[:, :, joint.value] = generateHeatmap(64, 1,
                      [vertical[FLIC.JOINT_TO_INDEX[joint]] * scale,
                      (horizontal[FLIC.JOINT_TO_INDEX[joint]] + pad['left']
                       - center['horizontal'] + resolution['vertical']//2) * scale])
        return heatmaps
    
    
    ''' Calculate PCK threshold.
    
    Arg:
        index: Data index.
        
    Return:
        PCK threshold
    '''
    def __getThresholdPCK(self, index):
        return 64
    
    
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
    
    
    ''' Get mask of FLIC.
    
    Return:
        Binary list representing mask of FLIC.
    '''
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
    def __squeeze(annotation, indices):
        if len(indices) == 0:
            return np.squeeze(annotation)
        return FLIC.__squeeze(np.squeeze(annotation)[indices[0]], indices[1:])