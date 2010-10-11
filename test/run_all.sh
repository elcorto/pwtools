#!/bin/bash

# Simple script to run all test_*.py . 
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

logfile=run_all.log
rm -f $logfile

# This is for test_f2py_flib_openmp.py
if env | grep OMP_NUM_THREADS > /dev/null; then
    old_omp=$OMP_NUM_THREADS
fi    
export OMP_NUM_THREADS=3

rm -vf $(find ../ -name "*.pyc")  $(find . -name "*.pyc") 2>&1 \
    | tee -a $logfile

[ -z $PYTHON ] && PYTHON=python
echo "using python interpreter: $PYTHON" | tee -a $logfile

# Simple-minded way to run test suite. Yes, we really should use nose.
for f in $(ls -1 *.py); do 
    echo "#################################################################"
    echo "$f"
    echo "#################################################################"
    $PYTHON $f
done 2>&1 | tee -a $logfile

if egrep -i 'error|warning' run_all.log > /dev/null; then
    echo "#################################################################"
    echo "there have been errors/warnings"
    echo "#################################################################"
fi

if [ -n "$old_omp" ]; then
    export OMP_NUM_THREADS=$old_omp
else
    unset OMP_NUM_THREADS
fi

./check.sh
