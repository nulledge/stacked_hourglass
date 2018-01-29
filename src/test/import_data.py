import imageio
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath('..'))

from data import DataCenter
from data_impl.FLIC import FLIC
from data_impl.MPII import MPII

reader = DataCenter(root=os.path.expanduser('~/Workspace/data/')) \
    .request(data='MPII', task='train', metric='PCKh')

assert MPII.NUMBER_OF_DATA == 25994 + 2889

batch = reader.getBatch(size=8)
for idx in range(8):
    imageio.imwrite('../img/image_MPII_' + str(idx) + '.jpg', batch[0][idx])
    for y in range(64):
        for x in range(64):
            batch[1][idx][y, x, 0] = max(batch[1][idx][y, x, :])
    imageio.imwrite('../img/heat_MPII_' + str(idx) + '.jpg', batch[1][idx][:, :, 0])
'''

one_epoch = int(MPII.NUMBER_OF_DATA * MPII.TRAIN_RATIO) * 2
cnt = 0
reader.reset()
test_iter = tqdm(total=one_epoch)
for _ in range(one_epoch):
    batch = reader.getBatch(size=8)
    cnt = cnt + batch[0].shape[0]
    test_iter.update(batch[0].shape[0])
test_iter.close()

assert cnt == int(MPII.NUMBER_OF_DATA * MPII.TRAIN_RATIO) * 2



reader = DataCenter(root=os.path.expanduser('~/Workspace/data/')) \
    .request(data='FLIC', task='train', metric='PCK')

assert FLIC.NUMBER_OF_DATA == 5003

batch = reader.getBatch(size=8)
for idx in range(8):
    imageio.imwrite('../img/image_FLIC_' + str(idx) + '.jpg', batch[0][idx])
    for y in range(64):
        for x in range(64):
            batch[1][idx][y, x, 0] = max(batch[1][idx][y, x, :])
    imageio.imwrite('../img/heat_FLIC_' + str(idx) + '.jpg', batch[1][idx][:, :, 0])
one_epoch = int(FLIC.NUMBER_OF_DATA * FLIC.TRAIN_RATIO) * 2
cnt = 0
reader.reset()
test_iter = tqdm(total=one_epoch)
for _ in range(one_epoch):
    batch = reader.getBatch(size=8)
    cnt = cnt + batch[0].shape[0]
    test_iter.update(batch[0].shape[0])
test_iter.close()

assert cnt == int(FLIC.NUMBER_OF_DATA * FLIC.TRAIN_RATIO) * 2
'''

print('complete.')