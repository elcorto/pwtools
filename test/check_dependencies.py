#!/usr/bin/env python3

import glob
import importlib
import json
import os
import subprocess
import re


def check_module(name):
    """Check if a package/module is installed.

    First try to import it. This works for most (e.g. numpy). If that fails,
    check "pip list". Need this for packages where package name != import name,
    such as

        * PyCifRW: import CifFile
        * pytest-xdist: import xdist
        * scikit-learn: import sklearn

    Our system pip (Debian) lists also system packages not installed by pip
    such that we get a list of all Python packages on the system. But we
    don't trust pip on other systems, so trying to import first and use "pip
    list" as a fallback is better.

    Convert dashes to underscores, such that e.g.

        # requirements: pytest-timeout
        python3 -c "import pytest_timeout"

    works.
    """
    try:
        name_lower = name.replace("-", "_")
        importlib.import_module(name_lower)
        print(f"  {name:20} ... ok (import)")
    except ImportError:
        if name in pip_list:
            print(f"  {name:20} ... ok (pip list)")
        else:
            print(f"  {name:20} ... NOT FOUND")


def backtick(call):
    """Version of pwtools.common.backtick() with return code check. Replicate
    here in order to not import from pwtools when checking for
    dependencies before install. Relative import may also not work if
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


def read_req_file(fn):
    """Read package names from requirements file. Skip comments.

    For now also skip things like

        git+https://github.com/elcorto/sphinx-autodoc
        git+https://github.com/elcorto/sphinx-autodoc@master

    (easy to support, but corner case, add if needed)
    """
    pkg_names = []
    rex_skip = re.compile(r"(^\s*#|^\s*$|.*github.*|.*gitlab.*)")
    with open(fn) as fd:
        for line in fd.readlines():
            name = line.strip()
            if name != "" and rex_skip.search(name) is None:
                pkg_names.append(name)
    return pkg_names



if __name__ == '__main__':

    # pip has no Python API, so ... yeah. This is slow, do it only once. Yes
    # this is a global var, but hey the script is tiny.
    pip_list = [x['name'] for x in
                json.loads(backtick("pip list --format=json 2> /dev/null"))]

    path = '../'
    req_files = glob.fnmatch.filter(os.listdir(path), "requirements*.txt")
    for name in req_files:
        print(name)
        req_filename = f'{path}/{name}'
        for pkg in read_req_file(req_filename):
            check_module(pkg)

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
