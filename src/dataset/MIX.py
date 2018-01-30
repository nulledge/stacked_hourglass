from .FLIC import FLIC
from .MPII import MPII
from . import DataCenter

class MIX:
    def __init__(self, root, task, metric):
        self.__root = root
        self.__task = task
        self.__metric = metric

        self.__FLIC = DataCenter(root=root).request(data='FLIC', task=task, metric='PCK')
        self.__MPII = DataCenter(root=root).request(data='MPII', task=task, metric='PCKh')

    def __delete__(self):
        pass

    def reset(self):
        self.__FLIC.reset()
        self.__MPII.reset()

    def getBatch(self, size):
        pass