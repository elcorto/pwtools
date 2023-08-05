# This module defines a dir where tests can write their temp files. It must be
# imported by any test module which needs a temp dir.
#
# This file is only a dummy default fallback in case someone runs a test in
# /path/to/pwtools/test (pytest test_foo.py). This file gets overwritten in a
# safe location when tests are run by runtests.sh.
testdir='/tmp'
