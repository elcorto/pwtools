# utils.py
#
# Math-related utils.
#

import numpy as np
from math import acos, pi


#-----------------------------------------------------------------------------

# Handles also complex arguments.
norm = np.linalg.norm
##def norm(a):
##    """2-norm for vectors."""
##    _assert(len(a.shape) == 1, "input must be 1d array")
##    # math.sqrt is faster then np.sqrt for scalar args
##    return math.sqrt(np.dot(a,a))

#-----------------------------------------------------------------------------

def angle(x,y):
    """Angle between vectors `x' and `y' in degrees."""
    # Numpy's `acos' is "acrcos", but wen take the one from math for scalar
    # args.
    return acos(np.dot(x,y)/norm(x)/norm(y))*180.0/pi
