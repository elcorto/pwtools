import numpy as np

from pwtools.structure import Structure
from pwtools import crys

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

    st = Structure(coords=coords,
                   cell=cell)
    np.testing.assert_array_almost_equal(coords_frac, st.get_coords_frac())

    st = Structure(coords_frac=coords_frac,
                   cell=cell)
    np.testing.assert_array_almost_equal(coords, st.get_coords())

    st = Structure(coords_frac=coords_frac,
                   cell=cell)
    np.testing.assert_array_almost_equal(crys_const, st.get_cryst_const())

    # Cell calculated from crys_const has defined orientation in space which may be
    # different from the original `cell`, but the volume and underlying crys_const
    # must be the same.
    try:
        st = Structure(coords_frac=coords_frac,
                       cryst_const=crys_const)
        np.testing.assert_array_almost_equal(cell, st.get_cell())
    except AssertionError:
        print "KNOWNFAIL: differrnt cell orientation"
    np.testing.assert_almost_equal(crys.volume_cell(cell),
                                   crys.volume_cell(st.get_cell()))
    np.testing.assert_array_almost_equal(crys_const, 
                                         crys.cell2cc(st.get_cell()))


    st = Structure(coords_frac=coords_frac,
                   cell=cell,
                   symbols=['X']*natoms)
    assert st.get_natoms() == natoms
    
    # Test if all getters work.
    st.attr_lst = ['coords',
                   'coords_frac',
                   'cell',
                   'cryst_const',
                   'natoms',
                   'atpos_str',
                   'celldm']
    for name in st.attr_lst:
        eval("st.get_%s()" %name)
    try:
        import ase
        st.get_ase_atoms()
    except ImportError:
        print("cannot import ase, skip test get_ase_atoms()")
