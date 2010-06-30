#!/bin/bash

# Simple script to run all test_*.py . For some tests, some plot windows
# may pop up. Just close them to continue.
#
# For tests calling Fortran extensions: Stdout from Fortran ends up in the
# wrong order in run_all.log. To see the correct output, run this these tests
# by hand (python test_foo.py).

# This is for test_f2py_flib_openmp.py
if env | grep OMP_NUM_THREADS > /dev/null; then
    old_omp=$OMP_NUM_THREADS
fi    
export OMP_NUM_THREADS=3

# Simple-minded way to run test suite. Yes, we really should use nose.
for f in $(ls -1 *.py); do 
    echo "#################################################################"
    echo "$f"
    echo "#################################################################"
    python $f
done 2>&1 | tee run_all.log

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
