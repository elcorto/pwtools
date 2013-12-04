#!/bin/bash

# ensure a clean generated tree
make clean
rm -rfv build/ source/generated/

# generate API doc rst files
sphinx-autodoc.py -s source -a generated/api -d generated/doc -w written \
      -X 'test\.test_' pwtools
