#!/bin/sh

loc=sphinx-autodoc/sphinx-autodoc.py
if which sphinx-autodoc.py; then 
    autodoc=sphinx-autodoc.py
elif [ -f $loc ]; then
    autodoc=$loc
else
    git clone https://github.com/elcorto/sphinx-autodoc
    autodoc=$loc
fi    

# ensure a clean generated tree
rm -v $(find ../ -name "*.pyc")
make clean
rm -rfv build/ source/generated/

# generate API doc rst files
echo "using: $autodoc"
$autodoc -s source -a generated/api \
         -X 'test\.test_|changelog|test\.check_dep' pwtools

# make heading the same level as in source/written/index.rst
sed -i -re '/^API.*/,/[-]+/ s/-/=/g' source/generated/api/index.rst
