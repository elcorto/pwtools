# version.py
#
# Utils to do version checking in scripts which use this package.
#
# The current implementation assumes that a file .hgtags resides in the
# package. This is true for the hg repo and for an archive created with "hg
# archive".
# 
# Examples
# --------
#
# >>> from pwtools import version as ver
# >>> assert ver.version > ver.tov('0.5.2'), "must use > 0.5.2"
# >>> assert ver.tov('0.5.2') < ver.version < ver.tov('0.6.0b1'), \
# "use version between 0.5.2 and 0.6.0b1"

from distutils.version import StrictVersion
import os.path 

def toversion(v1):
    return StrictVersion(v1)


def pathback(path, levels=1):
    p = path
    for i in range(levels):
        p = os.path.dirname(p)
    return p    

# alias 
tov = toversion

# version_str : The highest version number, e.g. "0.9.2" produced by "hg tags".
# It is the last tag before tip. This is the last one listed in .hgtags .
#     tip                              271:b733e197c96a
# --> 0.9.2                            268:a4ecac0432a8
#     0.9.1                            265:82817214abb2

hgtags = os.path.join(pathback(__file__, 1), '.hgtags')
if not os.path.exists(hgtags):
    print "warning: %s not found" %hgtags
    version_str = '0.0.0'
else:
    version_str = open(hgtags).readlines()[-1].split()[1]

version = toversion(version_str)
