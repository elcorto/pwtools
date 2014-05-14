from pwtools import symmetry, crys
from pwtools.test import tools
import numpy as np
import warnings

def test_symmetry():
    try: 
        from pyspglib import spglib
        st_prim = crys.Structure(
            coords_frac=np.array([[0]*3, [.5]*3]),
            cryst_const=np.array([3.5]*3 + [60]*3),
            symbols=['Al','N'])
        st_sc = crys.scell(st_prim,(2,3,4))
        st_prim2 = symmetry.spglib_get_primitive(st_sc, symprec=1e-2)
        # spglib returns (None,None,None) if given a primitive cell which
        # cannot be reduced any more
        assert symmetry.spglib_get_primitive(st_prim, symprec=1e-2) is None
        assert symmetry.spglib_get_primitive(st_prim2, symprec=1e-2) is None
        for st in [st_prim, st_sc, st_prim2]:
            assert symmetry.spglib_get_spacegroup(st_prim, symprec=1e-2) == (225, 'Fm-3m')
        tools.assert_dict_with_all_types_equal(st_prim.__dict__,
                                               st_prim2.__dict__,
                                               keys=['natoms',
                                                     'symbols',
                                                     'volume'])
    except ImportError:
        warnings.warn("WARNING: skipping test_symmetry, spglib not found")
