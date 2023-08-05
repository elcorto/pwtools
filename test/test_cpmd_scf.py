import os.path
from pwtools.parse import CpmdSCFOutputFile
from pwtools import common
from pwtools.test.tools import assert_attrs_not_none, unpack_compressed
pj = os.path.join

def test_cpmd_scf():
    filename = 'files/cpmd/scf/cpmd.out'
    basename = os.path.basename(filename)
    archive = os.path.dirname(filename) + '.tgz'
    workdir = unpack_compressed(archive)
    pp = CpmdSCFOutputFile(filename=pj(workdir, basename))
    pp.parse()
    assert_attrs_not_none(pp, none_attrs=[])
