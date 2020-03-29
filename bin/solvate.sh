#!/bin/bash

prog=$(basename $0)
logfile=${prog}.log
usage(){
cat << EOF
Add molecules (-ci option) to a molecule in a box (foo.pdb). If -ci is not
given, then water is used. We use Gromacs's "gmx insert-molecules" to insert.
Also we use pwtools's pdb2cif.py and cif2any.py scripts.

If you only have foo.cif or so, use openbabel's "babel" tool to convert first:

    $ apt-get install gromacs openbabel # Debian and derivatives
    $ babel foo.cif foo.pdb
    $ $prog foo.pdb ...

usage
-----
$prog foo.pdb [ -ci insert.pdb ] [<gmx-opts>]

will produce
  foo_gmx.pdb     # with water
  foo_gmx.cif     # with water, cif version
  foo_gmx.cif.txt # output from pwtools/bin/cif2any.py

options
-------
All options are passed to Gromacs, such as "-nmol" and "-seed". Run
    gmx help insert-molecules
for more options.

example
-------
$prog foo.pdb -nmol 18 -seed 123
$prog foo.pdb -nmol 18 -seed 123 -ci not_water.pdb
EOF
}

err(){
    echo "error: $@"
    exit 1
}

[ $# -eq 0 ] && err "no input args"
if echo "$@" | grep -qEe '-h|--help'; then
    usage
    exit 0
fi

if echo "$@" | grep -qEe '-ci '; then
    default_ci=
else
    default_ci="-ci water.pdb"
    cat > water.pdb << eof
ATOM      1  O  OSP3    1       4.013   0.831  -9.083  1.00  0.00
ATOM      2 1H  OSP3    1       4.941   0.844  -8.837  1.00  0.00
ATOM      3 2H  OSP3    1       3.750  -0.068  -9.293  1.00  0.00
TER
eof
fi

rm -f *.log \#*

name=${1/.pdb/}
shift
start_gmx=${name}.pdb
out_gmx=${name}_gmx.pdb
out_cif=${name}_gmx.cif

# add -ci molecule
# older Gromacs versions (< 5.x I guess) have a tool called genbox, with
# slighty different options, smth like
#   genbox_d -ci water.pdb -cp foo_gmx.pdb -nmol 18 -o foo_gmx.pdb
gmx insert-molecules $@ $default_ci -f $start_gmx -o $out_gmx >> $logfile 2>&1

pdb2cif.py $out_gmx $out_cif >> $logfile 2>&1
cif2any.py $out_cif > $out_cif.txt

grep -iE 'error|fatal|illegal' $logfile
grep 'Added.*molecules.*out of.*requested' $logfile
rm -f \#*
