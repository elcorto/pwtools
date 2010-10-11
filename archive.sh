#!/bin/bash

# Build version string and "hg archive". This only makes sense inside a
# Mercurial repository.

# We use some templates to fiddle out version information:
#
# latesttag: second-last tag, the one before tip
#     tip                              269:6a8b0f4e3c3d
#     0.9.2                            268:a4ecac0432a8  <<<<
#     0.9.1                            265:82817214abb2
# -> latesttag="0.9.2"
#
# latesttagdistance : The distance from tip to the latest tag "0.9.2" is 1.
#
#
# In the example above and "hg log -r tip --template
# '{latesttag}+{latesttagdistance}'", the version of tip would be "0.9.2+1".
# However, rev 269 is just created by tagging rev 268. So, in principle, the
# "real" version is rev 268 and one would write the version as "0.9.2" or
# "0.9.2+0". For this, one would either always subtract 1 from
# $latesttagdistance or simply archive rev 268 instead of tip.
#
##latesttag=$(hg log -r tip --template '{latesttag}')
##latesttagdistance=$(hg log -r tip --template '{latesttagdistance}')
##version=${latesttag}+$(echo "$latesttagdistance - 1" | bc)
#
# We just name the archive pwtools-{latesttag} and that's it. The
# {latesttagdistance} and the repo hashes are stored in the archive in
# .hg_archival.txt, e.g.
#    repo: 74d7fa88bdc3ecd15a1dc103a6c7ed35692c90fd
#    node: 085de072fc20df52ac60415603911e0a8bc2c81c
#    branch: default
#    latesttag: 0.9.2
#    latesttagdistance: 1
#
version=$(hg log -r tip --template '{latesttag}')
hg archive -t tgz pwtools-${version}.tgz
