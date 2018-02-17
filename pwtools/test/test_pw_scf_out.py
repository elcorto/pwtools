import numpy as np
from pwtools.parse import PwSCFOutputFile
from pwtools import common, io
from pwtools.constants import Ang, Bohr
from pwtools.test.tools import aaae, assert_attrs_not_none, adae, \
    unpack_compressed 
from pwtools.test import tools    

def test_pw_scf_out():
    
    # ref data for Structure, all lengths in Ang, energy in eV
    natoms = 2
    symbols = ['Si', 'Si']
    cell = np.array([[-2.71536701,  0.        ,  2.71536701],
           [ 0.        ,  2.71536701,  2.71536701],
           [-2.71536701,  2.71536701,  0.        ]])
    forces = np.array([[ 2.57110316,  5.14220632,  7.71330948],
                       [-2.57110316, -5.14220632, -7.71330948]]) # eV / Ang
    nspecies = {'Si': 2}
    mass = np.array([ 28.0855,  28.0855]) # amu
    cryst_const = np.array([  3.84010885,   3.84010885,   3.84010885,  60. ,
            60.        ,  60.        ])
    symbols_unique = ['Si']
    etot = -258.58148870118305 # eV
    typat = [1, 1]
    volume = 40.041985843396688 # Ang**3
    stress = np.array([[ 9.825,   0.  ,   0.  ],
           [  0.  ,  9.825,   0.  ],
           [  0.  ,   0.  ,  9.825]]) # GPa
    coords_frac = np.array([[ 0.  ,  0.  ,  0.  ],
           [ 0.25,  0.25,  0.25]])
    pressure = 9.825 # GPa
    coords = np.array([[ 0.        ,  0.        ,  0.        ],
           [-1.35768351,  1.35768351,  1.35768351]])
    order = {'Si': 1}
    alat = 10.2626 # Bohr


    filename = 'files/pw.scf.out'
    common.system('gunzip %s.gz' %filename)

    # use_alat=False. Provide high-precision alat from outside (e.g.
    # from pw.in instead of parsing and using low-precision value from pw.out).
    # Here we use the same alat for the tests.
    pp1 = PwSCFOutputFile(filename=filename, 
                         use_alat=False, # alat=1.0
                         units={'length': alat*Bohr/Ang})
    struct1 = pp1.get_struct() # pp1.parse() called here
    assert_attrs_not_none(struct1) 
    assert_attrs_not_none(pp1) 
    assert pp1.scf_converged is True
    assert alat == pp1.get_alat(True)
    assert 1.0 == pp1.get_alat(False)
    
    aaae(cryst_const, struct1.cryst_const)
    aaae(cell, struct1.cell)
    aaae(coords, struct1.coords)
    aaae(coords_frac, struct1.coords_frac)
    aaae(forces, struct1.forces)
    aaae(stress, struct1.stress)
    assert np.allclose(volume, struct1.volume)
    assert np.allclose(etot, struct1.etot)
    assert np.allclose(pressure, struct1.pressure)
    
    # use_alat=True, alat = 10.2626 Bohr
    pp2 = PwSCFOutputFile(filename=filename, use_alat=True)
    struct2 = pp2.get_struct() # pp.parse() called here
    assert_attrs_not_none(struct2) 
    assert_attrs_not_none(pp2) 
    assert np.allclose(alat, pp2.alat)
    assert pp2.scf_converged is True
    assert alat == pp2.get_alat(True)    # Bohr
    assert 1.0 == pp2.get_alat(False)
    
    # Skip coords adn cell b/c they are modified by self.alat and
    # pp1.alat = 1.0, pp2.alat = 10.2626 
    attr_lst = common.pop_from_list(pp1.attr_lst, ['coords', 'cell'])
    adae(pp1.__dict__, pp2.__dict__, keys=attr_lst)         

    attr_lst = struct1.attr_lst
    adae(struct1.__dict__, struct2.__dict__, keys=attr_lst)         
    
    pp3 = PwSCFOutputFile(filename=filename)
    assert alat == pp3.get_alat() # self.use_alat=True default
    
    common.system('gzip %s' %filename)

def test_pw_scf_no_forces_stress():
    fn = unpack_compressed('files/pw.scf_no_forces_stress.out.gz')
    pp = PwSCFOutputFile(fn)
    assert pp.get_forces() is None
    assert pp.get_stress() is None
    st = io.read_pw_scf(fn)
    assert st.stress is None
    assert st.forces is None

def test_pw_scf_one_atom():
    fn = unpack_compressed('files/pw.scf_one_atom.out.gz')
    pp = PwSCFOutputFile(fn)
    pp.parse()
    not_none = [\
        'coords',
        'symbols',
        'stress',
        'etot',
        'forces',
        'nstep_scf',
        'cell',
        'natoms',
        'nkpoints',
    ]
    tools.assert_attrs_not_none(pp, attr_lst=not_none)
    assert pp.forces.shape == (1,3)
    assert pp.coords.shape == (1,3)
    assert pp.stress.shape == (3,3)
    assert pp.cell.shape == (3,3)
