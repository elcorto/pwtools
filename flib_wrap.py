from pwtools.num import distsq as _distsq
import warnings

def distsq(*args, **kwds):
    warnings.simplefilter("always")
    warnings.warn("flib_wrap will be deprecated. Use num.distsq()",
                  DeprecationWarning)
    return _distsq(*args, **kwds)
