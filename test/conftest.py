import tempfile
import pathlib
import os
import datetime

import pytest

from pwtools.test.testenv import testdir


# Intermediate way to transition all tests to using this fixture instead of
# the testenv hack directly, while still keeping the mechanics in runtests.sh
# unchanged, for now.
#
# Additional feature: each pwtools_tmpdir is unique by construction, which
# should give us "thread-safe" dirs for free, i.e. for using pytest-xdist, w/o
# the need to create random dirs inside tests.
#
# "request" is a pytest fixture that holds information about the fixture's
# requesting scope, such as the module and the requesting test function inside
# that, and much more. Cool!
#
@pytest.fixture
def pwtools_tmpdir(request):
    base_path = os.path.join(
        testdir,
        "pwtools-test",
        request.module.__name__,
        request.function.__name__,
    )
    os.makedirs(base_path, exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H.%M.%SZ")
    path = tempfile.mkdtemp(dir=base_path, prefix=f"{stamp}_")
    return pathlib.Path(path)
