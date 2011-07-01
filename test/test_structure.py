import numpy as np

from pwtools.structure import Structure
from pwtools import crys, constants
from pwtools.test.tools import aaae

def test():
    natoms = 10
    cell = np.array([[3,0,0],
                     [1.1,5,-0.04],
                     [-0.33,1.5,7]])
    crys_const = crys.cell2cc(cell)                 
    coords_frac = np.random.rand(natoms,3)
    coords = crys.coord_trans(coords=coords_frac,
                              old=cell,
                              new=np.identity(3),
                              align='rows')
    symbols = ['H']*natoms

    st = Structure(coords=coords,
                   symbols=symbols,
                   cell=cell)
    aaae(coords_frac, st.get_coords_frac())

    st = Structure(coords_frac=coords_frac,
                   symbols=symbols,
                   cell=cell)
    aaae(coords, st.get_coords())

    st = Structure(coords_frac=coords_frac,
                   symbols=symbols,
                   cell=cell)
    aaae(crys_const, st.get_cryst_const())

    # Cell calculated from crys_const has defined orientation in space which may be
    # different from the original `cell`, but the volume and underlying crys_const
    # must be the same.
    try:
        st = Structure(coords_frac=coords_frac,
                       symbols=symbols,
                       cryst_const=crys_const)
        aaae(cell, st.get_cell())
    except AssertionError:
        print "KNOWNFAIL: differrnt cell orientation"
    np.testing.assert_almost_equal(crys.volume_cell(cell),
                                   crys.volume_cell(st.get_cell()))
    aaae(crys_const, crys.cell2cc(st.get_cell()))


    st = Structure(coords_frac=coords_frac,
                   symbols=symbols,
                   cell=cell)
    assert st.get_natoms() == natoms
    
    # Test if all getters work.
    for name in st.attr_lst:
        if name != 'ase_atoms':
            print name
            st.check_get_attr(name)
            assert getattr(st, name) is not None
    try:
        import ase
        atoms = st.get_ase_atoms()
        aaae(atoms.get_cell(), cell * constants.Bohr / constants.Angstrom)

    except ImportError:
        print("cannot import ase, skip test get_ase_atoms()")
