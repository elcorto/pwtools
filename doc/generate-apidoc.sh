#!/bin/bash

# try to find local checkout of sphinx-autodoc, else assume that it is on PATH
if [ -f sphinx-autodoc/sphinx-autodoc.py ]; then
    autodoc=sphinx-autodoc/sphinx-autodoc.py
else
    autodoc=sphinx-autodoc.py
fi    

# ensure a clean generated tree
rm -v $(find ../ -name "*.pyc")
make clean
rm -rfv build/ source/generated/

# generate API doc rst files
echo "using: $autodoc"
$autodoc -s source -a generated/api \
         -X 'test\.test_|changelog|test\.check_depend' pwtools
