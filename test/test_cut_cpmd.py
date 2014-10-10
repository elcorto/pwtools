import os.path
from pwtools.parse import CpmdMDOutputFile
from pwtools import common
from pwtools.test.tools import unpack_compressed

def run(dr, none_attrs=[]):
    dr = dr[:-1] if dr.endswith('/') else dr
    archive = dr + '.tgz'
    workdir = unpack_compressed(archive)
    common.system('../bin/cut-cpmd.sh %s 20 > %s/cut-cpmd.log' %(workdir,
        workdir))

def test_cut_cpmd():
    run(dr='files/cpmd/md_cp_pr')

