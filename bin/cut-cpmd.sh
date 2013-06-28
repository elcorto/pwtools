#!/bin/bash

prog=$(basename $0)
simulate=false
sed_opts="-r -e"
inplace=true
bakname=bak

usage(){
    cat << EOF

Truncate CPMD output files from the <step>'s iteration. Which files is
hardcoded:

CELL
ENERGIES
STRESS
TRAJECTORY
FTRAJECTORY
TRAJEC.xyz

Files which don't exist will be skipped.

usage:
------
$prog [-hs] <dir> <step>

options:
--------
-s : simulate
--not-inplace : don't use ``sed -i.$bakname ...``, use if YKWYAD 

example
-------
Say your calc ran 8423 steps, then was killed. The restart file is written
every 50 steps, so the last was written at 8400. A restart using "RESTART
ACCUMULATORS" and continuing files with "<<< NEW DATA >>>" markers starts at
8401 and thus repeats 8401..8423. You want to delete these 23 steps from the
old CELL, ENERGIES, etc files before the restart, such that they are cleanly
continued. Then use:

$ $prog /path/to/calc/ 8401 
EOF
}

msg(){
    echo "$prog: $@"
}    

err(){
    echo "$prog: error: $@"
    exit 1
}    

cmdline=$(getopt -o hs -l not-inplace -- "$@")
eval set -- "$cmdline"
while [ $# -gt 0 ]; do
    case "$1" in 
        -s)
            simulate=true
            ;;
        --not-inplace)
            inplace=false
            ;;
        -h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "cmdline error"
            exit 1
            ;;
    esac
    shift
done

[ $# -ne 2 ] && err "illegal number of args (need 2)"
dr=$1
[ -d "$dr" ] || err "not a dir: $dr"
step=$2

while read conf; do
    fn=$(echo "$conf" | awk -F '@@' '{print $1}')
    sedstr=$(echo "$conf" | awk -F '@@' '{print $2}')
    _bakname=$(echo "$conf" | awk -F '@@' '{print $3}')
    [ -n "$_bakname" ] && bakname=$_bakname
    if $inplace; then
        cmd="sed -i.$bakname $sed_opts '$sedstr' $fn"
    else        
        cmd="sed $sed_opts '$sedstr' $fn"
    fi        
    if [ -f $fn ]; then
        echo ">> $fn <<"
        echo "  $cmd"
        $simulate || eval $cmd
    else
        msg "not found: $fn"
    fi        
done << EOF
$dr/CELL@@/.*CELL PARAMETERS.*Step.*$step.*/,$ d
$dr/ENERGIES@@/^\\\s+$step\\\s.*/,$ d
$dr/STRESS@@/.*TOTAL STRESS.*Step.*$step.*/,$ d
$dr/TRAJECTORY@@/^\\\s+$step\\\s.*/,$ d
$dr/FTRAJECTORY@@/^\\\s+$step\\\s.*/,$ d
$dr/TRAJEC.xyz@@/^\\\s+STEP:\\\s+$step.*/,$ d
$dr/TRAJEC.xyz@@$,$ d@@bak2
EOF
