from pwtools import common as co
import tempfile, os
from .testenv import testdir

def test_makedirs():
    tmpdir = tempfile.mkdtemp(dir=testdir, prefix=__file__)
    tgt = os.path.join(tmpdir, 'foo')
    co.makedirs(tgt)
    assert os.path.exists(tgt)
    # pass
    co.makedirs('')
