# version.py
#
# Utils to do version checking in scripts which use this package.
#
# The current implementation assumes that a file .hgtags resides in the
# package. This is true for the hg repo and for an archive created with "hg
# archive".
# 
# version_str : The highest version number, e.g. "0.22.0" produced by "hg tags".
#     $ hg tags
#     tip                              568:07c8c4fe5724  
#     0.22.0                           567:73644e353af2  <---
#     0.21.2                           564:0e355f991b7c
#     ...
# 
# It is the last tag before tip and the last one listed in .hgtags .
# 
# Note that when using mercurial queues, we have this:
#     $ hg tags
#     tip                              569:07c8c4fe5724
#     qtip                             569:07c8c4fe5724
#     qbase                            569:07c8c4fe5724
#     doc-update.patch                 569:07c8c4fe5724
#     qparent                          568:604fcb00b735
#     0.22.0                           567:73644e353af2 <---
#     0.21.2                           564:0e355f991b7c
#     ...
# 
# But still, the last "real" tag (0.22.0 in the example) is the last one in
# .hgtags .
# 
# examples:
# ---------
# >>> from pwtools import version as ver
# >>> assert ver.version > ver.tov('0.5.2'), "must use > 0.5.2"
# >>> assert ver.tov('0.5.2') < ver.version < ver.tov('0.6.0b1'), \
# "use version between 0.5.2 and 0.6.0b1"

import warnings
from distutils.version import StrictVersion
import os.path 

def toversion(v1):
    return StrictVersion(v1)


def pathback(path, levels=1):
    p = path
    for i in range(levels):
        p = os.path.dirname(p)
    return p    

hgtags = os.path.join(pathback(__file__, 1), '.hgtags')
if not os.path.exists(hgtags):
    warnings.warn("file not found: %s, using version_str='0.0.0'" %hgtags)
    version_str = '0.0.0'
else:
    version_str = open(hgtags).readlines()[-1].split()[1]

version = toversion(version_str)

# alias 
tov = toversion
