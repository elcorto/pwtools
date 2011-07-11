#!/bin/bash

# Build version string and "hg archive". This only makes sense inside a
# Mercurial repository.
#
# We use some templates to fiddle out version information. 
# If latesttagdistance == 1, we use 
#     version={latesttag} 
#     example: version=0.9.2
# else    
#     version={latesttag}-{rev}-{node|short}
#     example: version=0.9.2-269-6a8b0f4e3c3d
# 
# Example: (call "hg tags")
#     tip                              269:6a8b0f4e3c3d
#     0.9.2                            268:a4ecac0432a8
#     0.9.1                            265:82817214abb2
# -> latesttag="0.9.2"
# -> latesttagdistance=1
#
# latesttag: second-last tag, the one before tip
# latesttagdistance: The distance from tip to the latest tag "0.9.2". In the
# example, latesttagdistance==1 b/c rev269 is only created by tagging rev268.
# "hg archive" creates a file ".hg_archival.txt", which looks like that: 
#
#    repo: 74d7fa88bdc3ecd15a1dc103a6c7ed35692c90fd
#    node: 6a8b0f4e3c3ddf52ac60415603911e0a8bc2c81c
#    branch: default
#    latesttag: 0.9.2
#    latesttagdistance: 1

dist=$(hg log -r tip --template '{latesttagdistance}')
if [ "$dist" == "1" ]; then
    version=$(hg log -r tip --template '{latesttag}')
else
    version=$(hg log -r tip --template '{latesttag}-{rev}-{node|short}')
fi
hg archive -t tgz pwtools-${version}.tgz
