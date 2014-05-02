import tempfile
import numpy as np
from pwtools.parse import PwMDOutputFile
from pwtools import common, parse
from pwtools.test.tools import assert_attrs_not_none 
from pwtools.test.testenv import testdir

def test_pw_vc_relax_out():
    filename = 'files/pw.vc_relax.out'
    common.system('gunzip %s.gz' %filename)
    pp = PwMDOutputFile(filename=filename)
    pp.parse()
    common.system('gzip %s' %filename)
    none_attrs = ['coords', 
                  'ekin', 
                  'temperature',
                  'timestep',
                  ]
    assert_attrs_not_none(pp, none_attrs=none_attrs)
    traj = pp.get_traj()
    none_attrs = [\
        'ekin', 
        'temperature',
        'timestep',
        'velocity',
        'time',
        ]
    assert_attrs_not_none(traj, none_attrs=none_attrs)   


# for test_return_3d_if_no_cell_unit
_cell = parse.traj_from_txt("""
   1.004152675   0.000000000   0.000000000
  -0.502076337   0.869621726   0.000000000
   0.000000000   0.000000000   1.609289155
   1.004147458   0.000000000   0.000000000
  -0.502073729   0.869617208   0.000000000
   0.000000000   0.000000000   1.609759673
   1.004050225   0.000000000   0.000000000
  -0.502025112   0.869533001   0.000000000
   0.000000000   0.000000000   1.610320650
   1.003992235   0.000000000   0.000000000
  -0.501996117   0.869482780   0.000000000
   0.000000000   0.000000000   1.610416170
   1.003981055   0.000000000   0.000000000
  -0.501990527   0.869473099   0.000000000
   0.000000000   0.000000000   1.610369398
   1.003981055   0.000000000   0.000000000
  -0.501990527   0.869473099   0.000000000
   0.000000000   0.000000000   1.610369398
""", shape=(6,3,3))

def test_return_3d_if_no_cell_unit():
    tmpdir = tempfile.mkdtemp(dir=testdir, prefix=__file__)
    base = 'pw.vc_relax_no_cell_unit.out'
    filename = '{tdr}/{base}'.format(tdr=tmpdir, base=base)
    cmd = "mkdir -p {tdr}; cp files/{base}.gz {tdr}/; \
           gunzip {fn}.gz;".format(tdr=tmpdir,
                                   base=base, fn=filename)
    common.system(cmd, wait=True)
    pp = PwMDOutputFile(filename=filename)
    pp.parse()
    assert np.allclose(pp.cell, _cell*pp.get_alat())
