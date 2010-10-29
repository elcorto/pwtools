#!/bin/bash

# Simple script to run all test_*.py . This is a very simple-minded way to run
# test suite. Yes, we really should use nose.
#
# For tests calling Fortran extensions: Stdout from Fortran ends up in the
# wrong order in run_all.log. To see the correct output, run these tests
# by hand (python test_foo.py).
#
# The default interpreter is "python". If you want to test another python
# version, use the PYTHON env var.
#
# usage:
#   [ PYTHON=pythonX.Y ] ./run_all.sh


# Make sure that the correct (this) package is picked up by the interpreter,
# no matter how you named it (e.g. "from pwtools import *" will fail
# if the package's root dir is named /path/to/pwtools-dev or such). For
# security, we copy the whole package to a tmp dir and run the tests there.
PP=$PYTHONPATH
tgt=$(cd .. && pwd)
testdir=/tmp/pwtools-test.$$
mkdir -pv $testdir
logfile=$testdir/run_all.log
cp -rv $tgt $testdir/pwtools | tee -a $logfile
export PYTHONPATH=$testdir:$PYTHONPATH
cd $testdir/pwtools/
# build extension modules
[ -f Makefile ] && make
cd test/

echo "Running tests ..." | tee -a $logfile

# This is for test_f2py_flib_openmp.py
if env | grep OMP_NUM_THREADS > /dev/null; then
    old_omp=$OMP_NUM_THREADS
fi    
export OMP_NUM_THREADS=3

rm -vf $(find ../ -name "*.pyc")  $(find . -name "*.pyc") 2>&1 \
    | tee -a $logfile

[ -z $PYTHON ] && PYTHON=python
echo "using python interpreter: $PYTHON" | tee -a $logfile

for f in $(ls -1 *.py); do 
    echo "#################################################################"
    echo "$f"
    echo "#################################################################"
    $PYTHON $f
done 2>&1 | tee -a $logfile

if egrep -i 'error|warning' $logfile > /dev/null; then
    echo "#################################################################"
    echo "there have been errors/warnings"
    echo "#################################################################"
fi

if [ -n "$old_omp" ]; then
    export OMP_NUM_THREADS=$old_omp
else
    unset OMP_NUM_THREADS
fi

export PYTHONPATH=$PP
echo "#################################################################"
echo "error/warning summary:"
echo "#################################################################"
./log_summary.sh $logfile
echo "#################################################################"
echo "Logfile: $logfile"
echo "#################################################################"
