#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import yoloparser

try:
    from PIL import Image
    from PIL import ImageDraw
    import pycuda.driver as cuda
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have pycuda and the example dependencies installed. 
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

try:
    import uff
except ImportError:
    raise ImportError("""Please install the UFF Toolkit""")

try:
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have the TensorRT Library installed 
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1)
