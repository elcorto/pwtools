import numpy as np
import itertools
from pwtools import crys, common, atomic_data, num
from pwtools.crys import Structure, Trajectory
from pwtools.test import tools
rand = np.random.rand

syms = itertools.cycle(atomic_data.symbols[1:])

def test_scell():
    cell = np.identity(3)
    coords_frac = np.array([[0.5, 0.5, 0.5],
                       [1,1,1]])
    symbols = ['Al', 'N']
    sc = crys.scell(Structure(coords_frac=coords_frac,
                              cell=cell,
                              symbols=symbols), (2,2,2))

    sc_coords_frac = \
        np.array([[ 0.25,  0.25,  0.25],
                  [ 0.25,  0.25,  0.75],
                  [ 0.25,  0.75,  0.25],
                  [ 0.25,  0.75,  0.75],
                  [ 0.75,  0.25,  0.25],
                  [ 0.75,  0.25,  0.75],
                  [ 0.75,  0.75,  0.25],
                  [ 0.75,  0.75,  0.75],
                  [ 0.5 ,  0.5 ,  0.5 ],
                  [ 0.5 ,  0.5 ,  1.  ],
                  [ 0.5 ,  1.  ,  0.5 ],
                  [ 0.5 ,  1.  ,  1.  ],
                  [ 1.  ,  0.5 ,  0.5 ],
                  [ 1.  ,  0.5 ,  1.  ],
                  [ 1.  ,  1.  ,  0.5 ],
                  [ 1.  ,  1.  ,  1.  ]])

    sc_symbols = ['Al']*8 + ['N']*8
    sc_cell = \
        np.array([[ 2.,  0.,  0.],
                  [ 0.,  2.,  0.],
                  [ 0.,  0.,  2.]])

    assert sc.symbols == sc_symbols
    np.testing.assert_array_almost_equal(sc.coords_frac, sc_coords_frac)
    np.testing.assert_array_almost_equal(sc.cell, sc_cell)

    # non-orthorhombic cell
    cell = \
        np.array([[ 1.,  0.5,  0.5],
                  [ 0.25,  1.,  0.2],
                  [ 0.2,  0.5,  1.]])

    sc = crys.scell(Structure(coords_frac=coords_frac,
                              cell=cell,
                              symbols=symbols), (2,2,2))
    sc_cell = \
        np.array([[ 2. ,  1. ,  1. ],
                  [ 0.5,  2. ,  0.4],
                  [ 0.4,  1. ,  2. ]])
    np.testing.assert_array_almost_equal(sc.cell, sc_cell)
    # crystal coords are cell-independent
    np.testing.assert_array_almost_equal(sc.coords_frac, sc_coords_frac)


    # slab
    #
    # Test if old and new implementation behave for a tricky case: natoms == 2
    # mask.shape[0], i.e. if reshape() behaves correctly.
    # Reference generated with old implementation. Default is new.
    cell = np.identity(3)
    coords_frac = np.array([[0.5, 0.5, 0.5],
                       [1,1,1]])
    symbols = ['Al', 'N']
    sc = crys.scell(Structure(coords_frac=coords_frac,
                              cell=cell,
                              symbols=symbols), (1,1,2))
    sc_coords_frac = \
        np.array([[ 0.5 ,  0.5 ,  0.25],
                  [ 0.5 ,  0.5 ,  0.75],
                  [ 1.  ,  1.  ,  0.5 ],
                  [ 1.  ,  1.  ,  1.  ]])
    sc_cell = \
        np.array([[ 1.,  0.,  0.],
                  [ 0.,  1.,  0.],
                  [ 0.,  0.,  2.]])
    sc_symbols = ['Al', 'Al', 'N', 'N']
    assert sc.symbols == sc_symbols
    np.testing.assert_array_almost_equal(sc.cell, sc_cell)
    np.testing.assert_array_almost_equal(sc.coords_frac, sc_coords_frac)

    sc = crys.scell(Structure(coords_frac=coords_frac,
                              cell=cell,
                              symbols=symbols), (1,2,1))
    sc_coords_frac = \
        np.array([[ 0.5 ,  0.25,  0.5 ],
                  [ 0.5 ,  0.75,  0.5 ],
                  [ 1.  ,  0.5 ,  1.  ],
                  [ 1.  ,  1.  ,  1.  ]])
    sc_cell = \
        np.array([[ 1.,  0.,  0.],
                  [ 0.,  2.,  0.],
                  [ 0.,  0.,  1.]])
    assert sc.symbols == sc_symbols
    np.testing.assert_array_almost_equal(sc.cell, sc_cell)
    np.testing.assert_array_almost_equal(sc.coords_frac, sc_coords_frac)

    sc = crys.scell(Structure(coords_frac=coords_frac,
                              cell=cell,
                              symbols=symbols), (2,1,1))
    sc_coords_frac = \
        np.array([[ 0.25,  0.5 ,  0.5 ],
                  [ 0.75,  0.5 ,  0.5 ],
                  [ 0.5 ,  1.  ,  1.  ],
                  [ 1.  ,  1.  ,  1.  ]])
    sc_cell = \
        np.array([[ 2.,  0.,  0.],
                  [ 0.,  1.,  0.],
                  [ 0.,  0.,  1.]])
    assert sc.symbols == sc_symbols
    np.testing.assert_array_almost_equal(sc.cell, sc_cell)
    np.testing.assert_array_almost_equal(sc.coords_frac, sc_coords_frac)

    # symbols = None
    sc = crys.scell(Structure(coords_frac=coords_frac,
                              cell=cell,
                              symbols=None), (2,2,2))
    assert sc.symbols is None

    # Trajectory
    natoms = 4
    nstep = 100
    symbols = [next(syms) for ii in range(natoms)]
    # cell 2d
    coords_frac = rand(nstep,natoms,3)
    cell = rand(3,3)
    dims = (2,3,4)
    nmask = np.prod(dims)
    sc = crys.scell(Trajectory(coords_frac=coords_frac,
                               cell=cell,
                               symbols=symbols),
                      dims=dims)
    assert sc.coords_frac.shape == (nstep, nmask*natoms, 3)
    assert sc.symbols == common.flatten([[sym]*nmask for sym in symbols])
    assert sc.cell.shape == (nstep,3,3)
    np.testing.assert_array_almost_equal(sc.cell,
                                         num.extend_array(cell * np.asarray(dims)[:,None],
                                                          sc.nstep,
                                                          axis=0))
    # cell 3d
    cell = rand(nstep,3,3)
    sc = crys.scell(Trajectory(coords_frac=coords_frac,
                               cell=cell,
                               symbols=symbols),
                      dims=dims)
    assert sc.coords_frac.shape == (nstep, nmask*natoms, 3)
    coords_frac2 = np.array([crys.scell(Structure(coords_frac=coords_frac[ii,...,],
                                                  cell=cell[ii,...],
                                                  symbols=symbols), dims=dims).coords_frac \
                             for ii in range(nstep)])
    np.testing.assert_array_almost_equal(sc.coords_frac, coords_frac2)
    assert sc.symbols == common.flatten([[sym]*nmask for sym in symbols])
    assert sc.cell.shape == (nstep,3,3)
    np.testing.assert_array_almost_equal(sc.cell,
                                         cell * np.asarray(dims)[None,:,None])

    # methods
    natoms = 20
    coords_frac = rand(natoms,3)
    cell = rand(3,3)
    dims = (2,3,4)
    symbols = [next(syms) for ii in range(natoms)]
    struct = Structure(coords_frac=coords_frac,
                       cell=cell,
                       symbols=symbols)
    sc1 = crys.scell(struct, dims=dims, method=1)
    sc2 = crys.scell(struct, dims=dims, method=2)
    d1 = dict([(key, getattr(sc1, key)) for key in sc1.attr_lst])
    d2 = dict([(key, getattr(sc2, key)) for key in sc2.attr_lst])
    tools.assert_dict_with_all_types_almost_equal(d1, d2)


def test_direc_and_neg_dims():
    for dims in [(2,3,4), (-2,-3,-4), (-2,3,4), (2,-3,4), (2,3,-4)]:
        m1 = crys.scell_mask(*dims, direc=1)[::-1,:]
        m2 = crys.scell_mask(*dims, direc=-1)
        assert (m1==m2).all()

    for direc in [1,-1]:
        ref = crys.scell_mask( 2, 3, 4, direc=direc)
        assert ((ref + crys.scell_mask(-2,-3,-4, direc=direc)) == 0.0).all()

        now = crys.scell_mask(-2, 3, 4, direc=direc)
        assert ((ref[:,0] + now[:,0]) == 0.0).all()
        assert (ref[:,1:] == now[:,1:]).all()

        now = crys.scell_mask(2, -3, 4, direc=direc)
        assert ((ref[:,1] + now[:,1]) == 0.0).all()
        assert (ref[:,(0,2)] == now[:,(0,2)]).all()

        now = crys.scell_mask(2, 3, -4, direc=direc)
        assert ((ref[:,2] + now[:,2]) == 0.0).all()
        assert (ref[:,:2] == now[:,:2]).all()

    natoms = 20
    coords_frac = rand(natoms,3)
    cell = rand(3,3)
    dims = (2,3,4)
    symbols = [next(syms) for ii in range(natoms)]
    struct = Structure(coords_frac=coords_frac,
                       cell=cell,
                       symbols=symbols)
    # coords are offset by a constant shift if all dims and direc are < 0
    s1 = crys.scell(struct, (2,3,4), direc=1)
    s2 = crys.scell(struct, (-2,-3,-4), direc=-1)
    d = s1.coords - s2.coords
    assert np.abs(d - d[0,:][None,:]).sum() < 1e-12

