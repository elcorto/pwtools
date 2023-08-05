#!/bin/sh

# This script does
#
# * testenv hack
# * copy package to testdir
# * run pytest there
#
# testenv hack
# ------------
# We communicate the $testdir variable to test_*.py modules by writing it to
# $testenv_module before running pytest. All tests which write temp files must
# import this module and write their files to "testdir":
#
#   >>> from pwtools.test.testenv import testdir
#   >>> filename = os.path.join(testdir, 'foo_tmp.txt')
#   >>> ...
#
# We should update to better temp dir tools such as pytest's tmp_path fixture
# or the stdlib tempfile.TemporaryDirectory in all tests. Then the testenv hack
# can be removed.
#
# copy package
# ------------
# ATM many tests still write temp data as they please, e.g. into a relative path
# next to the test script. Until then we keep this script and copy the package
# to $testdir first in order to not pollute this repo with test artifact files.

set -u

usage(){
    cat << EOF
Prepare package for testing and call pytest.

In particular, copy source to a temp location and run tests there. This gives
us a clean build environment.

Usage
-----
./runtests.sh [-h | --nobuild] [pytest options]

Options
-------
--nobuild : don't build extension modules, just copy already complied *.so
    files if present (run "make" before if not)

Examples
--------
./runtests.sh -v
Be a bit more verbose by calling pytest -v .

./runtests.sh -vs
Print all stdout of all tests as well (pytest -vs).

./runtests.sh test_foo.py test_bar.py
Run only some tests.

./runtests.sh --nobuild test_foo.py test_bar.py
Run these w/o building extensions b/c you know that they haven't changed.

./runtests.sh --nobuild --numprocesses=4 --timeout=60
Multiprocessing, yeah! May need --timeout on slow machines or if many more
processes than cores are used. pytest plugins needed for this:
-n/--numprocesses: pytest-xdist, --timeout: pytest-timeout.

Notes
-----
* For tests calling Fortran extensions: Stdout from Fortran ends up in the
  wrong order in the logfile. To see the correct output, run these tests by
  hand (./runtests.sh test_foo.py).

* We make sure that the correct (this) package is picked up by the interpreter,
  no matter how you named it (e.g. "from pwtools import *" will fail if the
  package's root dir is named /path/to/pwtools-dev or such). Therefore, and b/c
  of security, we copy the whole package to a tmp dir and run the tests there.

* For test_f2py_flib_openmp.py, we set OMP_NUM_THREADS=3. This will
  oversubscribe any CPU with less than 3 cores, but should run fine.

* test_rbf.py may occasionally fail if the generated random data is not good. Just
  re-run the test in that case.
EOF
}

prnt(){
    echo "$@" | tee -a $logfile
}

err(){
    echo "error: $@"
    exit 1
}

here=$(pwd)
scriptdir=$(readlink -f $(dirname $0))

# Simple cmd line parsing. Found no way to pass $@, which can contain
# pytest options + other (--nobuild), thru getopt(1) w/o it complaining
# about invalid options.
if echo "$@" | egrep -qe "-h|--help"; then
    usage
    exit 0
fi
if echo "$@" | egrep -qe "--nobuild"; then
    build=false
    params=$(echo "$@" | sed -re 's/--nobuild//g')
else
    build=true
    params=$@
fi

runner_opts="$params --color=yes"
for runner in pytest pytest-3; do
    if which $runner > /dev/null; then
        found_runner=true
        break
    fi
done
$found_runner || err "no pytest runner found"

# /path/to/pwtools
repo_root=$(realpath $scriptdir/../)

testdir=/tmp/pwtools-test.$$
testenv_module=$testdir/pwtools/src/pwtools/test/testenv.py
mkdir -pv $testdir
logfile=$testdir/runtests.log
rsync_excl=$testdir/_rsync.excl

prnt "copy package ..."
cat > $rsync_excl << EOF
.hg/
.git/
*.pyc
*.pyo
*.pyf
**/__pycache__
doc/
.*_cache
EOF
$build && echo '**.so' >> $rsync_excl
rsync -av $repo_root $testdir --exclude-from=$rsync_excl > $logfile 2>&1
prnt "... ready"

if $build; then
    prnt "build extension modules ..."
    cd $testdir/src/_ext_src/ && make gfortran-omp -B >> $logfile && cd $here 2>&1
    prnt "... ready"
fi

cd $testdir/pwtools/test

[ -e $testenv_module ] || err "$testenv_module doesn't exist"
echo "testdir='$testdir'" > $testenv_module

# logfile: We could use another pytest plugin
# https://pypi.org/project/pytest-reportlog/ to write a logfile. But in order
# to not bloat the list of dependencies, we use shell tools instead. When
# pytest is used in a pipe, it disables terminal color output. However we want
# colored output there, so we need to use pytest --color=yes (default is
# --color=auto). Then we just remove the color sequences from the logfile.
prnt "running tests ..."

add_to_pp=$testdir/pwtools/src
[ -n "${PYTHONPATH:-}" ] && pp=$add_to_pp:$PYTHONPATH || pp=$add_to_pp
eval "PYTHONPATH=$pp OMP_NUM_THREADS=3 $runner \
      $runner_opts" 2>&1 | tee -a $logfile

# Remove color sequences.
sed -i -re "s/[[:cntrl:]]\[[0-9]{1,3}m//g" $logfile

prnt "... ready"

echo "logfile: $logfile"
