from pwtools import crys
from pwtools.test import tools
import numpy as np
rand = np.random.rand

def test_get_ase_atoms():
    natoms = 10
    st = crys.Structure(coords_frac=rand(natoms,3),
                        symbols=['H']*10,
                        cell=rand(3,3))
    try:
        import ase
        st2 = crys.atoms2struct(crys.struct2atoms(st))
        keys = ['natoms', 'coords', 'coords_frac', 'symbols', 'cryst_const',
                'cell', 'volume', 'mass']
        tools.assert_dict_with_all_types_almost_equal(\
            st.__dict__,
            st2.__dict__,
            keys=keys,
            strict=True)
        # in case the test fails, use this to find out which key failed            
##        for kk in keys:
##            print("testing: %s ..." %kk)
##            tools.assert_all_types_almost_equal(st.__dict__[kk], st2.__dict__[kk])
        for pbc in [True,False]:
            at = st.get_ase_atoms(pbc=pbc)
            assert (at.pbc == np.array([pbc]*3)).all()
            at = crys.struct2atoms(st, pbc=pbc)
            assert (at.pbc == np.array([pbc]*3)).all()
    except ImportError:
        tools.skip("cannot import ase, skipping test get_ase_atoms()")
