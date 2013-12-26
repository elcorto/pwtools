#!/usr/bin/env python

"""
Load Fortran extensions and print all function signatures.
"""

# XXX maybe use that to auto-generate an rst file with API docs for the
# extension functions.

def print_doc_attr(module):
    name = "module = " + module.__name__
    fn   = "file   = " + module.__file__
    bar = "="*79
    print "%s\n%s\n%s\n%s" %(bar, name, fn, bar)
    for key,val in module.__dict__.iteritems():
        if not key.startswith('__') and hasattr(val, '__doc__'):
            doc = getattr(val, '__doc__')
            print doc

from pwtools import _flib
print_doc_attr(_flib)
from pwtools import _dcd
print_doc_attr(_dcd)
