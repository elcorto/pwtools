#!/usr/bin/env python3

import os, importlib

def check_module(name):
    try:
        importlib.import_module(name)
        print("    %s ... ok" %name)
    except ImportError:
        print("    %s ... NOT FOUND" %name)

if __name__ == '__main__':

    print("required packages:")
    for name in ['numpy', 'scipy', 'ase', 'spglib', 'CifFile', 'h5py',
                 'nose', 'matplotlib', 'sphinx', 'numpydoc']:
        check_module(name)

    print("optional executables:")
    for name in ['eos.x']:
        ok = False
        for path in os.environ['PATH'].split(':'):
            if os.path.exists("%s/eos.x" %path):
                print("    %s ... ok" %name)
                ok = True
                break
        if not ok:
            print("    %s ... NOT FOUND" %name)

