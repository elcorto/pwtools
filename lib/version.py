# version.py
#
# Simple utils to do version checking. It is up to the user to do that.
# 
# Examples
# --------
#
# >>> from pwtools import version as ver
#
# >>> assert ver.version > ver.tov('0.5.2'), "must use > 0.5.2"
# 
# >>> assert ver.tov('0.5.2') < ver.version < ver.tov('0.6.0b1'), \
# "use version between 0.5.2 and 0.6.0b1"

from distutils.version import StrictVersion

def toversion(v1):
    return StrictVersion(v1)

# alias 
tov = toversion

# The highest version number produced by "hg tags".
version = toversion("0.9.2")
