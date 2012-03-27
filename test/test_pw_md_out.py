from pwtools.parse import PwMDOutputFile
from pwtools import common
from pwtools.constants import Bohr, Ang
from pwtools.test.tools import assert_attrs_not_none, ade

def test_pw_md_out():    
    filename = 'files/pw.md.out'
    common.system('gunzip %s.gz' %filename)
    alat = 5.9098 # Bohr
    pp = PwMDOutputFile(filename=filename, use_alat=True)
    pp.parse()
    none_attrs = [\
        'coords_frac',
        ]
    assert_attrs_not_none(pp, none_attrs=none_attrs)
    traj = pp.get_traj()
    assert_attrs_not_none(traj)

    pp1 = pp
    traj1 = traj
    pp2 = PwMDOutputFile(filename=filename, 
                         use_alat=False,
                         units={'length': alat*Bohr/Ang})
    pp2.parse()                         
    assert_attrs_not_none(pp2, none_attrs=none_attrs)
    ade(pp1.__dict__, pp2.__dict__, attr_lst=pp1.attr_lst)
    traj2 = pp2.get_traj()
    ade(traj1.__dict__, traj2.__dict__, attr_lst=traj1.attr_lst)

    pp3 = PwMDOutputFile(filename=filename)
    assert alat == pp3.get_alat() # self.use_alat=True default
   
    common.system('gzip %s' %filename)
