from pwtools.parse import PwMDOutputFile
from pwtools import common
from pwtools.test.tools import assert_attrs_not_none 

def test_pw_vc_relax_out():
    filename = 'files/pw.vc-relax.out'
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
        ]
    assert_attrs_not_none(traj, none_attrs=none_attrs)   
