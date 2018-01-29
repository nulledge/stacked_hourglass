import os

from src.data_impl.FLIC import FLIC
from src.data_impl.MPII import MPII

''' Common interface to combine data, task and metric.
'''


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

    def request(self, data, task, metric):
        path = os.path.join(self.__root, data)

        if data == 'FLIC':
            return FLIC(root=path, task=task, metric=metric)
        elif data == 'MPII':
            return MPII(root=path, task=task, metric=metric)
        else:
            raise NotImplementedError()
