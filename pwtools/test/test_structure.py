import types
import numpy as np

from pwtools.crys import Structure, Trajectory
from pwtools import crys, constants
from pwtools.test.tools import aaae, assert_all_types_equal
from pwtools.test import tools
from pwtools.test.utils.rand_container import get_rand_struct
rand = np.random.rand

# We assume all lengths in Angstrom. Only importans for ASE comparison.

def test_struct():
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
        print(name)
        st.try_set_attr(name)
        assert getattr(st, name) is not None, "attr None: %s" %name
        assert eval('st.get_%s()'%name) is not None, "getter returns None: %s" %name
    aaae(coords_frac, st.coords_frac)
    aaae(cryst_const, st.cryst_const)
    aaae(coords, st.coords)
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
        if val is not None and not (isinstance(val, int) or \
            isinstance(val, float)):
            val2 = getattr(st2,name)
            assert id(val2) != id(val)
            assert_all_types_equal(val2, val)


def test_get_traj():
    st = Structure(coords_frac=rand(20,3),
                   symbols=['H']*20,
                   forces=rand(20,3),
                   stress=rand(3,3),
                   etot=42.23)
    nstep = 5
    tr = st.get_traj(nstep)
    for name in st.attr_lst:
        print(name)
        attr = getattr(tr, name)
        if attr is not None:
            if name in tr.attrs_nstep:
                assert attr.shape[tr.timeaxis] == nstep
            else:
                attr_st = getattr(st, name)
                assert_all_types_equal(attr, attr_st)

def test_coord_trans():
    natoms = 10
    cell = np.array([[3,0,0],
                     [1.1,5,-0.04],
                     [-0.33,1.5,7]])
    cryst_const = crys.cell2cc(cell)
    coords_frac = rand(natoms,3)
    coords = crys.coord_trans(coords=coords_frac,
                              old=cell,
                              new=np.identity(3))

    st = crys.Structure(coords=coords,
                        cell=cell)
    assert np.allclose(coords_frac, st.coords_frac)
    st = crys.Structure(coords_frac=coords_frac,
                        cell=cell)
    assert np.allclose(coords, st.coords)

    st = crys.Structure(coords=coords,
                        cell=cell)
    assert np.allclose(cryst_const, st.cryst_const)


def test_get_fake_ase_atoms():
    st = get_rand_struct()
    atoms = st.get_fake_ase_atoms()
    assert (st.coords_frac == atoms.get_scaled_positions()).all()
    assert (st.cell == atoms.get_cell()).all()
    assert (atoms.get_atomic_numbers() == np.array(st.get_znucl())).all()


def test_znucl():
    st = Structure(symbols=['Al']*2 + ['N']*3)
    assert st.znucl == [13]*2 + [7]*3
    assert st.znucl_unique == [13,7]

def test_mix():
    symbols = ['H']*3 + ['Au']*7
    natoms = len(symbols)
    st1nf = Structure(coords_frac=rand(natoms,3),
                    symbols=symbols,
                    cell=rand(3,3))
    st2nf = Structure(coords_frac=rand(natoms,3),
                    symbols=symbols,
                    cell=rand(3,3))
    st1f = Structure(coords_frac=rand(natoms,3),
                    symbols=symbols,
                    cell=rand(3,3),
                    forces=rand(natoms,3))
    st2f = Structure(coords_frac=rand(natoms,3),
                    symbols=symbols,
                    cell=rand(3,3),
                    forces=rand(natoms,3))
    for st1,st2 in [(st1f, st2f), (st1nf, st2nf)]:
        tr = crys.mix(st1, st2, alpha=np.linspace(0,1,20))
        assert tr.nstep == 20
        assert tr.coords_frac.shape == (20, st1.natoms, 3)

        for idx,st in [(0,st1), (-1, st2)]:
            tools.assert_dict_with_all_types_almost_equal(st.__dict__,
                                                          tr[idx].__dict__,
                                                          keys=st1.attr_lst)
        for x in [0.5, 0.9]:
            tr = crys.mix(st1, st2, alpha=np.array([x]))
            assert tr.nstep == 1
            tools.assert_all_types_almost_equal(tr[0].coords, (1-x)*st1.coords + x*st2.coords)
            tools.assert_all_types_almost_equal(tr[0].cell, (1-x)*st1.cell + x*st2.cell)
            if tr.forces is not None:
                tools.assert_all_types_almost_equal(tr[0].forces, (1-x)*st1.forces + x*st2.forces)


