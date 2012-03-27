import os.path
from pwtools.parse import CpmdSCFOutputFile
from pwtools import common
from pwtools.test.tools import assert_attrs_not_none
pj = os.path.join

def test():
    basedr = 'files/cpmd'
    dr = 'files/cpmd/scf'
    common.system('tar -C %s -xzf %s.tgz' %(basedr, dr))
    filename = os.path.join(dr, 'cpmd.out')
    pp = CpmdSCFOutputFile(filename=filename)
    pp.parse()
    assert_attrs_not_none(pp, none_attrs=[])
    common.system('rm -r %s' %dr)
