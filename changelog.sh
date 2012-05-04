#!/bin/bash

# Call "hg log" and make a changelog separated by pattern in the
# commit message (API, ENH, BUG, ...).
# 
# usage:
# ------
# ./changelog.sh [-r <start>:<end>]
#
# examples
# --------
# ./changelog.py
# ./changelog.py -r 0.30.1:tip
#
# notes:
# ------
# We have the convention to format commit messages like that:
#
# ENH: foo bar
#      baz
# BUG: foo bar
#      baz
# API: foo bar
#      baz
# INT: foo bar
#      baz
#
# where ENH = enhancement, BUG = bug fix, API = API change, INT = internal
# refactoring. If a commit message doesn't fit into that pattern, it will be
# ignored. If you want to saerch the history by yourself, then clone the repo
# and use "hg log".

dbar="==============================================================================="
sbar="-------------------------------------------------------------------------------"

args=$(echo "$@" | sed -re 's/-r//')
[ -z "$args" ] && args="0:tip"
echo $args

for prefix in 'API' 'BUG' 'ENH'; do
    echo -e "\n$dbar"
    echo $prefix
    echo $dbar
    cmd="hg log -v -r \"$args and grep(r'^$prefix:')\" --template \"{desc}\\\n$sbar\\\n\""
##    echo -e "$cmd\n"
    eval $cmd
done    
