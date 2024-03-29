import os
import numpy as np
from pwtools.common import is_seq, file_write
from pwtools.test.testenv import testdir

def test_is_seq():
    fn = os.path.join(testdir, 'is_seq_test_file')
    file_write(fn, 'lala')
    fd = open(fn , 'r')
    for xx in ([1,2,3], (1,2,3), np.array([1,2,3])):
        print(type(xx))
        assert is_seq(xx) is True
    for xx in ('aaa', fd):
        print(type(xx))
        assert is_seq(xx) is False
    fd.close()
