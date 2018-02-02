import os

from ._flic import FLIC
from ._mpii import MPII
from ._mix import MIX


def getReader(path, name, batch_size):
    if name == 'FLIC':
        return FLIC(root=path, batch_size=batch_size)
    elif name == 'MPII':
        return MPII(root=path, batch_size=batch_size)
    elif name == 'MIX':
        return MIX(root=path, batch_size=batch_size, task='train')
