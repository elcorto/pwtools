import numpy as np
from pwtools.parse import PwSCFOutputFile
from pwtools import common
from pwtools.constants import Ang, Bohr
from pwtools.test.tools import aaae, aae, assert_attrs_not_none, ade

# TODO add fake forces in pw.scf.out and test unit conversion, right now forces
# are zero b/c the structure is perfect

def test():
    
    # ref data, all lengths in Ang
    natoms = 2
    symbols = ['Si', 'Si']
    cell = np.array([[-2.71536701,  0.        ,  2.71536701],
           [ 0.        ,  2.71536701,  2.71536701],
           [-2.71536701,  2.71536701,  0.        ]])
    forces = np.array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    nspecies = {'Si': 2}
    mass = np.array([ 28.0855,  28.0855])
    cryst_const = np.array([  3.84010885,   3.84010885,   3.84010885,  60. ,
            60.        ,  60.        ])
    symbols_unique = ['Si']
    etot = -258.58148870118305
    typat = [1, 1]
    volume = 40.041985843396688
    stress = np.array([[ 9.825,   0.  ,   0.  ],
           [  0.  ,  9.825,   0.  ],
           [  0.  ,   0.  ,  9.825]])
    coords_frac = np.array([[ 0.  ,  0.  ,  0.  ],
           [ 0.25,  0.25,  0.25]])
    pressure = 9.825
    coords = np.array([[ 0.        ,  0.        ,  0.        ],
           [-1.35768351,  1.35768351,  1.35768351]])
    order = {'Si': 1}
    alat = 10.2626 # Bohr


    filename = 'files/pw.scf.out'
    common.system('gunzip %s.gz' %filename)

    # use_alat=False. Provide high-precision alat from outside (e.g.
    # from pw.in instead of parsing and using low-precision value from pw.out).
    # Here we use the same alat for the tests.
    pp = PwSCFOutputFile(filename=filename, 
                         use_alat=False, # alat=1.0
                         units={'length': alat*Bohr/Ang})
    struct = pp.get_struct() # pp.parse() called here
    assert_attrs_not_none(struct) 
    assert_attrs_not_none(pp) 
    assert pp.scf_converged is True
    assert alat == pp.get_alat(True)
    assert 1.0 == pp.get_alat(False)
    
    aaae(cryst_const, struct.cryst_const)
    aaae(cell, struct.cell)
    aaae(coords, struct.coords)
    aaae(coords_frac, struct.coords_frac)
    aaae(forces, struct.forces)
    aaae(stress, struct.stress)
    aae(volume, struct.volume)
    aae(etot, struct.etot)
    aae(pressure, struct.pressure)
    
    pp1 = pp
    struct1 = struct

    # use_alat=True, alat = 10.2626 Bohr
    pp2 = PwSCFOutputFile(filename=filename, use_alat=True)
    struct2 = pp2.get_struct() # pp.parse() called here
    assert_attrs_not_none(struct2) 
    assert_attrs_not_none(pp2) 
    assert pp2.scf_converged is True
    assert alat == pp2.get_alat(True)    # Bohr
    assert 1.0 == pp2.get_alat(False)
    
    ade(pp1.__dict__, pp2.__dict__, attr_lst=pp1.attr_lst)         
    ade(struct1.__dict__, struct2.__dict__, attr_lst=struct1.attr_lst)         
    
    pp3 = PwSCFOutputFile(filename=filename)
    assert alat == pp3.get_alat() # self.use_alat=True default
    
    common.system('gzip %s' %filename)
