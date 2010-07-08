#!/bin/bash

# second-last tag, the one before tip
#     tip                              269:6a8b0f4e3c3d
#     0.9.2                            268:a4ecac0432a8  <<<<
#     0.9.1                            265:82817214abb2
# -> version="0.9.2"
version=$(hg tags | head -n2 | grep -v tip | awk '{print $1}')

# Archive tip, but name it "version". It is assumed that the difference between
# tip and the second-last changeset is just the tag operation, hence only
# .hgtags has changed.
hg archive -t tgz pwtools-${version}.tgz
