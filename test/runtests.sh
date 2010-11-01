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
#   Print all stdout (like warnings) of all tests.
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


tgt=$(cd .. && pwd)
testdir=/tmp/pwtools-test.$$
mkdir -pv $testdir
logfile=$testdir/runtests.log
cp -rvL $tgt $testdir/pwtools | tee -a $logfile
cd $testdir/pwtools/
# build extension modules
[ -f Makefile ] && make  | tee -a $logfile
cd test/

# Purge any compiled files.
echo "Deleting *.pyc files ..." | tee -a $logfile
rm -vf $(find ../ -name "*.pyc")  $(find . -name "*.pyc") 2>&1 \
    | tee -a $logfile

echo "Running tests ..." | tee -a $logfile
PYTHONPATH=$testdir:$PYTHONPATH \
OMP_NUM_THREADS=3 \
nosetests $@ 2>&1 | tee -a $logfile

cat << eof
#################################################################
Logfile error/warning summary. Logfile may not contain 
everything. Use $0 -s in that case. 
#################################################################
eof
egrep -i 'error|warning|fail' $logfile
cat << eof
#################################################################
Logfile: $logfile
#################################################################
eof
