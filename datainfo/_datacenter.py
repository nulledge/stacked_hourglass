import os

from ._flic import FLIC
from ._mpii import MPII
from ._lsp import LSP
from ._mix import MIX


def getReader(path, name, batch_size, task):
    if name == 'FLIC':
        return FLIC(root=path, batch_size=batch_size, task=task)
    elif name == 'MPII':
        return MPII(root=path, batch_size=batch_size, task=task)
    elif name == 'LSP':
        return LSP(root=path, batch_size=batch_size, task=task)
    elif name == 'MIX':
        return MIX(root=path, batch_size=batch_size, task='train')
