import types
import numpy as np

from pwtools.crys import Structure
from pwtools import crys, constants
from pwtools.test.tools import aaae, assert_all_types_equal
rand = np.random.rand

# We assume all lengths in Angstrom. Only importans for ASE comparison.

def test():
    natoms = 10
    cell = np.array([[3,0,0],
                     [1.1,5,-0.04],
                     [-0.33,1.5,7]])
    cryst_const = crys.cell2cc(cell)                 
    coords_frac = rand(natoms,3)
    coords = crys.coord_trans(coords=coords_frac,
                              old=cell,
                              new=np.identity(3))
    symbols = ['H']*natoms
    stress = rand(3,3)
    forces = rand(natoms,3)

    # Use ``cell`` instead of ``cryst_const` as input such that
    # atoms.get_cell() test passes (see below for why -- cell orientation)
    st = Structure(coords_frac=coords_frac,
                   symbols=symbols,
                   cell=cell,
                   stress=stress,
                   forces=forces,
                   etot=42)
    
    # Test if all getters work.
    for name in st.attr_lst:
        if name not in ['ase_atoms']:
            print name
            st.try_set_attr(name)
            assert getattr(st, name) is not None, "attr None: %s" %name
            assert eval('st.get_%s()'%name) is not None, "getter returns None: %s" %name
    try:
        import ase
        atoms = st.get_ase_atoms()
        aaae(atoms.get_cell(), cell)
    except ImportError:
        print("cannot import ase, skip test get_ase_atoms()")
    aaae(coords_frac, st.coords_frac)
    aaae(cryst_const, st.cryst_const)
    assert st.natoms == natoms

    st = Structure(coords_frac=coords_frac,
                   symbols=symbols,
                   cell=cell)
    aaae(coords, st.get_coords())

    # Cell calculated from cryst_const has defined orientation in space which may be
    # different from the original `cell`, but the volume and underlying cryst_const
    # must be the same.
    st = Structure(coords_frac=coords_frac,
                   symbols=symbols,
                   cryst_const=cryst_const)
    assert st.get_cell() is not None
    try:
        aaae(cell, st.get_cell())
    except AssertionError:
        print "KNOWNFAIL: differrnt cell orientation"
    np.testing.assert_almost_equal(crys.volume_cell(cell),
                                   crys.volume_cell(st.get_cell()))
    aaae(cryst_const, crys.cell2cc(st.get_cell()))

    # units
    st = Structure(coords_frac=coords_frac,
                    cell=cell,
                    symbols=symbols,
                    stress=stress,
                    forces=forces,
                    units={'length': 2, 'forces': 3, 'stress': 4})
    aaae(2*coords, st.coords)                    
    aaae(3*forces, st.forces)                    
    aaae(4*stress, st.stress)                    
    
    traj = crys.struct2traj(st)
    assert traj.is_traj

    # copy(): Assert everything has another memory location = is a new copy of
    # the object. IntTypes are NOT copied by copy.deepcopy(), which we use in
    # Structure.copy(), apparently b/c they are always automatically copied
    # before in-place operations. Same for float type. 
    #
    # >>> a=10; b=a; print id(a); print id(b)
    # 36669152
    # 36669152
    # >>> a*=100; print id(a); print id(b)
    # 72538264
    # 36669152
    # >>> a
    # 100
    # >>> b
    # 10
    #
    # >>> a=[1,2,3]; b=a; print id(a); print id(b)
    # 72624320
    # 72624320
    # >>> a[0] = 44; print id(a); print id(b)
    # 72624320
    # 72624320
    # >>> a
    # [44, 2, 3]
    # >>> b
    # [44, 2, 3]
    st2 = st.copy()
    for name in st.attr_lst:
        val = getattr(st,name)
        if val is not None and not (isinstance(val, types.IntType) or \
            isinstance(val, types.FloatType)):
            val2 = getattr(st2,name)
            assert id(val2) != id(val)
            assert_all_types_equal(val2, val)
