#!/usr/bin/env python3

import glob
import importlib
import json
import os
import subprocess


def check_module(name):
    """Check if a package/module is installed.

    First try to import it. This works for most (e.g. numpy). If that fails,
    check "pip list". Need this for packages where package name != import name,
    such as PyCifRW, where we need to "import CifFile"!

    Our system pip (Debian) lists also system packages not installed by pip
    such that we get a list of all Python packages on the system. But we
    don't trust pip on other systems, so trying to import first and use "pip
    list" as a fallback is better.
    """
    try:
        importlib.import_module(name)
        print(f"  {name:20} ... ok (import)")
    except ImportError:
        if name in pip_list:
            print(f"  {name:20} ... ok (pip list)")
        else:
            print(f"  {name:20} ... NOT FOUND")


def backtick(call):
    """Version of pwtools.common.backtick() with return code check. Replicate
    here in order to not import from pwtools here when checking for
    dependencies before install. Relative import may also not work of
    extensions are not yet built.

    Examples
    --------
    >>> print(backtick('ls -l'))
    """
    pp = subprocess.Popen(call, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = pp.communicate()
    if err.strip() != b'' or pp.returncode != 0:
        raise Exception(f"Error calling command, retcode={pp.returncode}"
                        f"\nStdout:\n{out.decode()}\nStderr:\n{err.decode()}")
    return out.decode()


if __name__ == '__main__':

    # pip has no Python API, so ... yeah. This is slow, do it only once. Yes
    # this is a global var, but hey the script is tiny.
    pip_list = [x['name'] for x in
                json.loads(backtick("pip list --format=json"))]

    path = '../../'
    req_files = glob.fnmatch.filter(os.listdir(path), "requirements*.txt")
    for name in req_files:
        print(name)
        with open(f'{path}/{name}') as fd:
            for pkg in fd.readlines():
                check_module(pkg.strip())

    print("optional executables:")
    for name in ['eos.x']:
        ok = False
        for path in os.environ['PATH'].split(':'):
            if os.path.exists(f"{path}/{name}"):
                print(f"  {name:20} ... ok")
                ok = True
                break
        if not ok:
            print(f"  {name:20} ... NOT FOUND")
