import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]

layer_path = os.path.join(script_path, 'layer')
sys.path.append(layer_path)

import tensorflow as tf

from conv import *
from bn import *
from relu import *
from pool import *