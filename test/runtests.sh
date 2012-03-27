#!/bin/bash

# Prepare package for testing and call nosetests.
#
# usage:
#   ./runtests.sh [nosetests options]
# 
# example:
#   ./runtests.sh -v
#   Be a bit more verbose by calling nosetests -v .
#   
#   ./runtests.sh -vs
#   Print all stdout (like warnings) of all tests (nosetests -vs).
#   
#   ./runtests.sh test_foo.py test_bar.py
#   Run only some tests.
#
# For tests calling Fortran extensions: Stdout from Fortran ends up in the
# wrong order in the logfile. To see the correct output, run these tests by
# hand (./runtests.sh test_foo.py).
# 
# We make sure that the correct (this) package is picked up by the interpreter,
# no matter how you named it (e.g. "from pwtools import *" will fail if the
# package's root dir is named /path/to/pwtools-dev or such). Therefore, and b/c
# of security, we copy the whole package to a tmp dir and run the tests there.
#
# For test_f2py_flib_openmp.py, we set OMP_NUM_THREADS=3. This will
# oversubscribe any CPU with less than 3 cores, but should run fine.

prnt(){
    echo "$@" | tee -a $logfile
}    

nose_opts="$@ --exclude='.*abinit.*'"
prnt "NOTE: All Abinit tests disabled!"

testdir=/tmp/pwtools-test.$$
mkdir -pv $testdir
logfile=$testdir/runtests.log
prnt "copy package ..."
tgt=$(cd .. && pwd)
cp -rvL $tgt $testdir/pwtools &>> $logfile
cd $testdir/pwtools/
prnt "... ready"
prnt "build extension modules ..."
[ -f Makefile ] && make &>> $logfile
prnt "... ready"
cd test/

# HACK: communicate variable to test_*.py modules. All tests which write temp
# files must import this module and write their files to "testdir":
# >>> from testenv import testdir
# >>> filename = os.path.join(testdir, 'foo_tmp.txt')
# >>> ...
echo "testdir='$testdir'" > testenv.py

# Purge any compiled files.
prnt 'deleting *.pyc files ...'
rm -vf $(find ../ -name "*.pyc")  $(find . -name "*.pyc") \
    &>> $logfile
prnt "... ready"

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
