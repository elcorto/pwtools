import numpy as np
from pwtools import crys, common
rand = np.random.rand

def test():
    cell = np.identity(3)
    coords = np.array([[0.5, 0.5, 0.5],
                       [1,1,1]])
    symbols = ['Al', 'N']
    sc = crys.scell(coords, cell, (2,2,2), symbols)

    sc_coords = \
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

    assert sc['symbols'] == sc_symbols
    np.testing.assert_array_almost_equal(sc['coords'], sc_coords)
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)
    
    # non-orthorhombic cell
    cell = \
        np.array([[ 1.,  0.5,  0.5],
                  [ 0.25,  1.,  0.2],
                  [ 0.2,  0.5,  1.]])

    sc = crys.scell(coords, cell, (2,2,2), symbols)
    sc_cell = \
        np.array([[ 2. ,  1. ,  1. ],
                  [ 0.5,  2. ,  0.4],
                  [ 0.4,  1. ,  2. ]])
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)
    # crystal coords are cell-independent
    np.testing.assert_array_almost_equal(sc['coords'], sc_coords)

    
    # slab
    #
    # Test if old and new implementation behave for a tricky case: natoms == 2
    # mask.shape[0], i.e. if reshape() behaves correctly in raw_scell().
    # Reference generated with old implementation. Default is new.
    cell = np.identity(3)
    coords = np.array([[0.5, 0.5, 0.5],
                       [1,1,1]])
    symbols = ['Al', 'N']
    sc = crys.scell(coords, cell, (1,1,2), symbols)
    sc_coords = \
        np.array([[ 0.5 ,  0.5 ,  0.25],
                  [ 0.5 ,  0.5 ,  0.75],
                  [ 1.  ,  1.  ,  0.5 ],
                  [ 1.  ,  1.  ,  1.  ]])
    sc_cell = \
        np.array([[ 1.,  0.,  0.],
                  [ 0.,  1.,  0.],
                  [ 0.,  0.,  2.]])
    sc_symbols = ['Al', 'Al', 'N', 'N']
    assert sc['symbols'] == sc_symbols
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)
    np.testing.assert_array_almost_equal(sc['coords'], sc_coords)
    
    sc = crys.scell(coords, cell, (1,2,1), symbols)
    sc_coords = \
        np.array([[ 0.5 ,  0.25,  0.5 ],
                  [ 0.5 ,  0.75,  0.5 ],
                  [ 1.  ,  0.5 ,  1.  ],
                  [ 1.  ,  1.  ,  1.  ]])
    sc_cell = \
        np.array([[ 1.,  0.,  0.],
                  [ 0.,  2.,  0.],
                  [ 0.,  0.,  1.]])
    assert sc['symbols'] == sc_symbols
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)
    np.testing.assert_array_almost_equal(sc['coords'], sc_coords)

    sc = crys.scell(coords, cell, (2,1,1), symbols)
    sc_coords = \
        np.array([[ 0.25,  0.5 ,  0.5 ],
                  [ 0.75,  0.5 ,  0.5 ],
                  [ 0.5 ,  1.  ,  1.  ],
                  [ 1.  ,  1.  ,  1.  ]])
    sc_cell = \
        np.array([[ 2.,  0.,  0.],
                  [ 0.,  1.,  0.],
                  [ 0.,  0.,  1.]])
    assert sc['symbols'] == sc_symbols
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)
    np.testing.assert_array_almost_equal(sc['coords'], sc_coords)
    
    # symbols = None
    sc = crys.scell(coords, cell, (1,1,1), symbols=None)
    assert sc['symbols'] is None
    
    # scell3d
    natoms = 4
    nstep = 100
    symbols = ['X%i' %idx for idx in range(natoms)]
    # cell 2d
    coords = rand(natoms, 3, nstep)
    cell = rand(3,3)
    dims = (2,3,4)
    nmask = np.prod(dims)
    sc = crys.scell3d(coords, cell, dims, symbols)
    assert sc['coords'].shape == (nmask*natoms, 3, nstep)
    assert sc['symbols'] == common.flatten([['X%i' %idx]*nmask for idx in \
                                            range(natoms)])
    assert sc['cell'].shape == (3,3)                                            
    np.testing.assert_array_almost_equal(sc['cell'], 
                                         cell * np.asarray(dims)[:,None])
    # cell 3d
    coords = rand(natoms, 3, nstep)
    cell = rand(3,3,nstep)
    dims = (2,3,4)
    nmask = np.prod(dims)
    sc = crys.scell3d(coords, cell, dims, symbols)
    assert sc['coords'].shape == (nmask*natoms, 3, nstep)
    assert sc['symbols'] == common.flatten([['X%i' %idx]*nmask for idx in \
                                            range(natoms)])
    assert sc['cell'].shape == (3,3,nstep) 
    np.testing.assert_array_almost_equal(sc['cell'], 
                                         cell * np.asarray(dims)[:,None,None])

