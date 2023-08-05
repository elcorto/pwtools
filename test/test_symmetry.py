import numpy as np

from pwtools import crys
from pwtools.test import tools
from pwtools.test.utils import rand_container as rc
from pwtools import symmetry


def test_struct2spglib():
    st = rc.get_rand_struct()
    tup1 = st.get_spglib()
    tup2 = symmetry.struct2spglib(st)
    assert tup1 == tup2
    assert len(tup1) == 3
    # cell
    assert tup1[0].shape == (3,3)
    # coords_frac
    assert tup1[1].shape == (st.natoms,3)
    assert (tup1[1] <= 1.0).all()
    # znucl
    assert tup1[2] == st.znucl


def test_struct2spglib_revert():
    st1 = rc.get_rand_struct()
    st2 = symmetry.spglib2struct(symmetry.struct2spglib(st1))
    # strict equality, only possible here b/c the conversion back and forth
    # must be perfect
    skip_attrs = [
        'etot',
        'forces',
        'pressure',
        'stress',
        ]
    attr_lst = set(st1.attr_lst) - set(skip_attrs)
    assert len(attr_lst) > 0
    for name in attr_lst:
        tools.assert_all_types_equal(getattr(st1, name),
                                     getattr(st2, name))


class TestIsSameStruct:
    def test_rand(self):
        st1 = rc.get_rand_struct()
        st2 = rc.get_rand_struct()
        assert not np.allclose(st1.volume, st2.volume)
        assert not np.allclose(st1.coords_frac, st2.coords_frac)
        assert not np.allclose(st1.cryst_const, st2.cryst_const)
        assert not symmetry.is_same_struct(st1, st2)

    def test_same(self):
        st1 = rc.get_rand_struct()
        assert symmetry.is_same_struct(st1, st1)

    def test_same_rotate_cell(self):
        st1 = rc.get_rand_struct()
        st2 = st1.copy()
        st2.cell = None
        st2.coords = None
        st2.set_all()
        assert not np.allclose(st1.coords, st2.coords)
        assert not np.allclose(st1.cell, st2.cell)
        assert symmetry.is_same_struct(st1, st1)

    def test_same_convert(self):
        st1 = rc.get_rand_struct()
        st2 = symmetry.spglib2struct(symmetry.struct2spglib(st1))
        assert symmetry.is_same_struct(st1, st2)


def test_symmetry_pwtools_api():
    st_prim = crys.Structure(
        coords_frac=np.array([[0]*3, [.5]*3]),
        cryst_const=np.array([3.5]*3 + [60]*3),
        symbols=['Al','N'])
    st_sc = crys.scell(st_prim, (2,3,4))
    st_prim2 = symmetry.spglib_get_primitive(st_sc, symprec=1e-2)
    # irreducible structs
    assert symmetry.spglib_get_primitive(st_prim, symprec=1e-2) is None
    assert symmetry.spglib_get_primitive(st_prim2, symprec=1e-2) is None
    for st in [st_prim, st_sc, st_prim2]:
        assert symmetry.spglib_get_spacegroup(st_prim, symprec=1e-2) == (225, 'Fm-3m')
    # this is redundant since we have is_same_struct(), but keep it anyway
    tools.assert_dict_with_all_types_almost_equal(st_prim.__dict__,
                                                  st_prim2.__dict__,
                                                  keys=['natoms',
                                                        'symbols',
                                                        'volume',
                                                        'cryst_const'])
    assert symmetry.is_same_struct(st_prim, st_prim2)
