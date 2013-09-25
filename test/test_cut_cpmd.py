import os.path
from pwtools.parse import CpmdMDOutputFile
from pwtools import common
from testenv import testdir

def run(dr, none_attrs=[]):
    # dr       = 'files/cpmd/md_bo'
    # basedr   = 'files/cpmd'
    # archive  = 'files/cpmd/md_bo.tgz'
    if dr.strip().endswith('/'):
        dr = dr.strip()[:-1]
    basedr = os.path.dirname(dr)
    common.system('tar -C %s -xzf %s.tgz' %(basedr, dr))
    common.system('../bin/cut-cpmd.sh %s 20 > %s/cut-cpmd.log' %(dr, testdir))
    common.system('rm -r %s' %dr)

def test_cut_cpmd():
    run(dr='files/cpmd/md_cp_pr/')

