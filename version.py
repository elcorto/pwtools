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
##from common import backtick
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
#     
##version_str = "0.9.2"
##version_str = backtick('hg tags | head -n2').split()[2]

hgtags = os.path.join(pathback(__file__, 1), '.hgtags')
if not os.path.exists(hgtags):
    print "warning: %s not found" %hgtags
    version_str = '0.0.0'
else:
    version_str = open(hgtags).readlines()[-1].split()[1]

version = toversion(version_str)
