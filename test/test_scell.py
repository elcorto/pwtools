import numpy as np
from pwtools import crys

def test():
    cp = np.identity(3)
    coords = np.array([[0.5, 0.5, 0.5],
                       [1,1,1]])
    symbols = ['Al', 'N']
    sc = crys.scell(coords, cp, (2,2,2), symbols)

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

    # slab
    sc = crys.scell(coords, cp, (1,1,3), symbols)
    sc_cell = \
        np.array([[ 1.,  0.,  0.],
                  [ 0.,  1.,  0.],
                  [ 0.,  0.,  3.]])
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)

    # non-orthorhombic cell
    cp = \
        np.array([[ 1.,  0.5,  0.5],
                  [ 0.25,  1.,  0.2],
                  [ 0.2,  0.5,  1.]])

    sc = crys.scell(coords, cp, (2,2,2), symbols)
    sc_cell = \
        np.array([[ 2. ,  1. ,  1. ],
                  [ 0.5,  2. ,  0.4],
                  [ 0.4,  1. ,  2. ]])
    np.testing.assert_array_almost_equal(sc['cell'], sc_cell)
    # crystal coords are cell-independent
    np.testing.assert_array_almost_equal(sc['coords'], sc_coords)
