import os.path
import subprocess as sp
from pwtools.test.tools import unpack_compressed

def run(dr):
    dr = dr[:-1] if dr.endswith('/') else dr
    workdir = unpack_compressed(dr + '.tgz')
    exe = os.path.join(os.path.dirname(__file__),
                       '../../bin/cut-cpmd.sh')
    cmd = '{e} {w} 20 > {w}/cut-cpmd.log'.format(e=exe, w=workdir)
    sp.run(cmd, check=True, shell=True)

def test_cut_cpmd():
    run(dr='files/cpmd/md_cp_pr')

