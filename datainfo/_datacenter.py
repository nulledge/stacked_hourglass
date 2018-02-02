import os

from ._flic import FLIC
from ._mpii import MPII
from ._mix import MIX


def getReader(path, name, batch_size):
    if name == 'FLIC':
        return FLIC(root=path, batch_size=batch_size)
    elif name == 'MPII':
        return MPII(root=path, batch_size=batch_size)

        # return MPII(root=path, task='train', metric='pck')
    elif name == 'MIX':
        return MIX(root=path, task='train')


class DataCenter(object):
    """ Set path to data.

    Args:
        root: relative path to root of data.
    """

    def __init__(self, root):
        self.__root = root

    ''' Combine data, task and metric.

    Args:
        data: 'MPII' or 'FLIC'.
        task: 'train' or 'eval'.
        metric: 'PCK' or 'PCKh'.
    '''

    def request(self, dataset_name, task, metric):
        path = os.path.join(self.__root)#, dataset_name)

        if dataset_name == 'FLIC':
            return FLIC(root=path, task=task, metric=metric)
        elif dataset_name == 'MPII':
            return MPII(root=path, task=task, metric=metric)
        elif dataset_name == 'MIX':
            return MIX(root=self.__root, task=task)
        else:
            raise NotImplementedError()
