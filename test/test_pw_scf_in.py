from pwtools.parse import PwInputFile
from pwtools import common, constants, crys
import numpy as np
asrt = np.testing.assert_array_almost_equal

# More thorough test than pw_scf_in_2. Here, we also do verification and test
# the feature that self.cell == CELL_PARAMETERS, wheres self.cryst_const is
# calculated using system:celldm(1) or system:A.

def test():
    filename = 'files/pw.scf.in'

    pp = PwInputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)

    none_attrs = []
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
    cell = np.array([[ 1.  ,  0.  ,  0.  ],
                     [ 0.  ,  1.  ,  0.  ],
                     [ 0.  ,  0.  ,  0.78]])
    alat = 10 # angstrom

    asrt(pp.cell, cell)
    asrt(pp.cryst_const, crys.cell2cc(cell*alat / constants.a0_to_A))
    asrt(pp.mass, np.array([  28.0855,   28.0855,  238.    ]))
    assert pp.symbols == ['Si', 'Si', 'U']
    asrt(pp.coords,
         np.array([[ 0.   ,  0.   ,  0.   ],
                   [ 0.25 ,  0.25 ,  0.25 ],
                   [-0.3  ,  0.5  ,  0.999]]))
    assert pp.kpoints == {'kpoints': '2 2 2 0 0 0', 'mode': 'automatic'}
    assert pp.atspec['symbols'] == ['Si', 'U']
    assert pp.atspec['pseudos'] == ['Si.pbe-n-van.UPF', 'U.no-working-PP.UPF']
    asrt(pp.atspec['masses'], np.array([  28.0855,  238.    ]))
    assert pp.natoms == 2
    assert pp.atpos['unit'] == 'crystal'
