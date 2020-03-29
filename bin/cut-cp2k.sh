#!/bin/bash

prog=$(basename $0)
simulate=false
bakname=bak

usage(){
    cat << EOF

Truncate cp2k output files from the <step>'s iteration to the end. Which files
is hardcoded:

PROJECT-1.cell
PROJECT-1.ener
PROJECT-1.stress
PROJECT-frc-1.xyz
PROJECT-pos-1.xyz
PROJECT-vel-1.xyz

Files which don't exist will be skipped.

PROJECT-1.restart is needed if <step> is not given. Then STEP_START_VAL+1 will
be used for <step>.

usage:
------
$prog [-hs] <dir> [<step>]

options:
--------
-s : simulate

example
-------
Say your calc ran 8423 steps, then was killed. The restart file is written
every 50 steps, so the last was written at 8400. A restart using
"EXT_RESTART%RESTART_COUNTERS T" starts at 8401 and thus repeats 8401..8423.
You want to delete these 23 steps from the old cell, ener, etc files before the
restart, such that they are cleanly continued. Then use:

$ $prog /path/to/calc/ 8401

If <step> is omited, then STEP_START_VAL+1 from the restart file will be used.
EOF
}

msg(){
    echo "$prog: $@"
}

err(){
    echo "$prog: error: $@"
    exit 1
}

cmdline=$(getopt -o hs -- "$@")
eval set -- "$cmdline"
while [ $# -gt 0 ]; do
    case "$1" in
        -s)
            simulate=true
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

dr=$1
[ -d "$dr" ] || err "not a dir: $dr"

# only to get STEP_START_VAL, the last finished time step where the restart
# file was written
file_restart=$dr/PROJECT-1.restart

# one step per line
file_cell=$dr/PROJECT-1.cell
file_ener=$dr/PROJECT-1.ener
file_stress=$dr/PROJECT-1.stress

# xyz file with (natoms,3) block per time step
file_frc=$dr/PROJECT-frc-1.xyz
file_pos=$dr/PROJECT-pos-1.xyz
file_vel=$dr/PROJECT-vel-1.xyz


if [ $# -eq 2 ]; then
    step_to_cut=$2
elif [ $# -eq 1 ]; then
    [ -f $file_restart ] || err "$file_restart not found, cannot determine step_to_cut"
    # last restart file written at step N
    last_step=$(awk '/STEP_START_VAL/ {print $2}' $file_restart | tr -d ' ')
    # cut all steps N+1 ... end, restart run starts with step N+1
    step_to_cut=$(echo "$last_step + 1" | bc)
else
    err "need 1 or 2 input args"
fi

echo "dir   =   $dr"
echo "step  =   $step_to_cut"

for fn in $file_cell $file_ener $file_stress; do
    if [ -f $fn ]; then
        cmd="sed -i.$bakname -re '/^\s+$step_to_cut\s+/,\$d' $fn"
        echo $fn
        $simulate && echo "    $cmd" || eval $cmd
    fi
done

natoms=$(head -n1 $file_pos | tr -d ' ')
for fn in $file_frc $file_pos $file_vel; do
    if [ -f $fn ]; then
        # Cut file, then delete last line, but only if it really contains only
        # one integer = $natoms.
        cmd="sed -i.$bakname -re '/i =\s+$step_to_cut,\s+/,\$d' $fn; \
             if tail -n1 $fn | egrep -q '[ ]+$natoms[ ]*\$'; then \
                 sed -i -re '\$d' $fn; \
             fi"
        echo $fn
        $simulate && echo "    $cmd" || eval $cmd
    fi
done

