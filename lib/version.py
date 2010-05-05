from distutils import version

# The highest version number produced by "hg tags".
current_version = "0.5.4"

def _convert(v1, v2):
    _v1 = version.StrictVersion(v1)
    _v2 = version.StrictVersion(v2)
    return _v1, _v2     

def greater_equal(v1, v2):
    _v1, _v2 = _convert(v1, v2)
    return _v1 >= _v2

def greater(v1, v2):
    _v1, _v2 = _convert(v1, v2)
    return _v1 > _v2

def lower_equal(v1, v2):
    _v1, _v2 = _convert(v1, v2)
    return _v1 <= _v2

def lower(v1, v2):
    _v1, _v2 = _convert(v1, v2)
    return _v1 < _v2

def require(v1, v2=current_version):
    if lower(v1, v2):
        raise StandardError("version %s required, %s given" %(v2, v1))
    else:
        pass
