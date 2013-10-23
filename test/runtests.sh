#!/bin/bash

usage(){
cat << EOF
Prepare package for testing and call nosetests.

Usage
-----
./runtests.sh [-h | --nobuild] [nosetests options]

Options
-------
--nobuild : don't build extension modules, just copy already complied *.so
    files if present (run "make" before if not)

Examples
--------
./runtests.sh -v
Be a bit more verbose by calling nosetests -v .

./runtests.sh -vs
Print all stdout (like warnings) of all tests (nosetests -vs).

./runtests.sh test_foo.py test_bar.py
Run only some tests.

./runtests.sh --nobuild test_foo.py test_bar.py
Run these w/o building extensions b/c you know that they haven't changed.

./runtests.sh --nobuild --processes=4 --process-timeout=60
Multiprocessing, yeah! May need --process-timeout on slow machines or if many
more processes than cores are used.

Notes
-----
* For tests calling Fortran extensions: Stdout from Fortran ends up in the
  wrong order in the logfile. To see the correct output, run these tests by
  hand (./runtests.sh test_foo.py).

* We make sure that the correct (this) package is picked up by the interpreter,
  no matter how you named it (e.g. "from pwtools import *" will fail if the
  package's root dir is named /path/to/pwtools-dev or such). Therefore, and b/c
  of security, we copy the whole package to a tmp dir and run the tests there.

* For test_f2py_flib_openmp.py, we set OMP_NUM_THREADS=3. This will
  oversubscribe any CPU with less than 3 cores, but should run fine.

* If all tests pass, good. If they do and the logfile does not contain warnings
  but only some KNOWNFAIL statements .. very good

* test_rbf.py may seldomly fail if the generated random data is not good. Just
  re-run the test in that case.
EOF
}

prnt(){
    echo "$@" | tee -a $logfile
}    

# Simple cmd line parsing. Found no way to pass $@, which can contain
# nosetests options + other (--nobuild), thru getopt(1) w/o it complaining
# about invalid options.
if echo "$@" | egrep -qe "-h|--help"; then
    usage
    exit 0
fi    
if echo "$@" | egrep -qe "--nobuild"; then 
    build=false
    params=$(echo "$@" | sed -re 's/--nobuild//g')
else
    build=true
    params=$@
fi    

nose_opts="$params" 

testdir=/tmp/pwtools-test.$$
tgtdir=$testdir/pwtools
mkdir -pv $tgtdir
logfile=$testdir/runtests.log

prnt "copy package ..."
rsync_excl=_rsync.excl
cat > $rsync_excl << EOF
.hg/
*.pyc
*.pyo
*.pyf
doc/
EOF
$build && echo '*.so' >> $rsync_excl
rsync -av ../ $tgtdir --exclude-from=$rsync_excl > $logfile 2>&1
rm $rsync_excl
cd $tgtdir
prnt "... ready"

if $build; then
    prnt "build extension modules ..."
    [ -f Makefile ] && make gfortran-omp -B >> $logfile 2>&1
    prnt "... ready"
fi 

cd test/

# HACK: communicate variable to test_*.py modules. All tests which write temp
# files must import this module and write their files to "testdir":
# >>> from testenv import testdir
# >>> filename = os.path.join(testdir, 'foo_tmp.txt')
# >>> ...
echo "testdir='$testdir'" > testenv.py

prnt "running tests ..."
PYTHONPATH=$testdir:$PYTHONPATH \
OMP_NUM_THREADS=3 \
eval "nosetests $nose_opts" 2>&1 | tee -a $logfile
prnt "... ready"

cat << eof

Logfile: $logfile
Logfile error/warning summary follows. Logfile may not contain everything. Use
$0 -s in that case. 
------------------------------------------------------------------------------
eof
egrep -i 'error|warning|fail' $logfile
cat << eof
------------------------------------------------------------------------------
eof
