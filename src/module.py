import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0]

module_path = os.path.join(script_path, 'module')
sys.path.append(module_path)

from bottleneck import *
from hourglass import *