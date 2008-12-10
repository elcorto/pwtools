#!/usr/bin/env python
# vim:ts=4:sw=4:et

# 
# Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
# The pydos package. 
# 

# Stolen from [1]. See also [2].
#
# usage:
#   
#  If build dir is present
#    $ rm -r build
#  
#  Build _flib.so locally, don't install anything
#    $ python setup.py build
#    $ cp build/lib.linux-i686-2.5/_flib.so .
#  
#  Test 
#    $ ./pydos.py -h
#  
#  Or use the Makefile :-)
#  
# [1] /path/to/scipy-sources/scipy/optimize/setup.py
# [2] http://projects.scipy.org/scipy/numpy/wiki/DistutilsDoc

from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package, top_path)

    sources = ['flib.pyf', 'flib.f90']
    config.add_extension('_flib', sources)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
