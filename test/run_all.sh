#!/bin/bash

# Simple-minded way to run test suite. Yes, we really should use node.
for f in $(ls -1 *.py); do 
    echo "#################################################################"
    echo "$f"
    echo "#################################################################"
    python $f
done 2>&1 | tee run_all.log

if egrep 'error|warning' run_all.log > /dev/null; then
    echo "#################################################################"
    echo "there have been errors/warnings"
    echo "#################################################################"
fi    
