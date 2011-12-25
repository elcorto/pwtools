from math import sqrt
import numpy as np
from pwtools.crys import coord_trans
rand = np.random.rand
asrt_almost_equal = np.testing.assert_array_almost_equal

def test():
    #-----------------------------------------------------------
    # 2D
    #-----------------------------------------------------------
    c_X = rand(20,3)
    # basis vecs are assumed to be rows
    X = rand(3,3)*5
    Y = rand(3,3)*3

    # transform and back-transform
    c_Y = coord_trans(c_X, old=X, new=Y)
    c_X2 = coord_trans(c_Y, old=Y, new=X)
    asrt_almost_equal(c_X, c_X2)

    # X and must have the right shape: (4,4) here
    try:
        coord_trans(rand(20,4), old=X, new=Y)
    except AssertionError:
        print "KNOWNFAIL"

    # simple dot product must produce same cartesian results:
    # X . v_X = I . v_I = v_I
    X = np.identity(3)
    Y = rand(3,3)*3
    c_X = rand(20,3)
    c_Y = coord_trans(c_X, old=X, new=Y)
    # normal back-transform
    c_X2 = coord_trans(c_Y, old=Y, new=X)
    # 2 forms w/ dot(), assume: basis vecs = rows of X and Y
    c_X3 = np.dot(c_Y, Y)
    c_X4 = np.dot(Y.T, c_Y.T).T
    asrt_almost_equal(c_X, c_X2)
    asrt_almost_equal(c_X, c_X3)
    asrt_almost_equal(c_X, c_X4)

    # some textbook example
    #
    v_I = np.array([1.0,1.5])
    I = np.identity(2)
    # basis vecs as rows
    X = sqrt(2)/2.0*np.array([[1,-1],[1,1]]).T
    Y = np.array([[1,1],[0,1]]).T

    # "identity" transform
    asrt_almost_equal(coord_trans(v_I,I,I), v_I)

    # v in basis X and Y
    v_X = coord_trans(v_I,I,X)
    v_Y = coord_trans(v_I,I,Y)
    asrt_almost_equal(v_X, np.array([1.76776695, 0.35355339]))
    asrt_almost_equal(v_Y, np.array([-0.5,  1.5]))

    # back-transform
    asrt_almost_equal(coord_trans(v_X,X,I), v_I)
    asrt_almost_equal(coord_trans(v_Y,Y,I), v_I)
    
    # higher "x,y,z"-dims: 4-vectors
    c_X = rand(20,4)
    X = rand(4,4)*5
    Y = rand(4,4)*3
    c_Y = coord_trans(c_X, old=X, new=Y)
    c_X2 = coord_trans(c_Y, old=Y, new=X)
    asrt_almost_equal(c_X, c_X2)
     
    
    #-----------------------------------------------------------
    # 3D
    #-----------------------------------------------------------

    # x,y,z case
    c_X = rand(20,3,10)
    X = rand(3,3)*5
    Y = rand(3,3)*3
    c_Y = coord_trans(c_X, old=X, new=Y, axis=1)
    c_X2 = coord_trans(c_Y, old=Y, new=X, axis=1)
    asrt_almost_equal(c_X, c_X2)
    
    c_X = rand(20,10,3)
    c_Y = coord_trans(c_X, old=X, new=Y, axis=-1)
    c_X2 = coord_trans(c_Y, old=Y, new=X, axis=-1)
    asrt_almost_equal(c_X, c_X2)

    c_X = rand(3,20,10)
    c_Y = coord_trans(c_X, old=X, new=Y, axis=0)
    c_X2 = coord_trans(c_Y, old=Y, new=X, axis=0)
    asrt_almost_equal(c_X, c_X2)
    
    # 3d, higher "x,y,z"-dims, i.e. 4-vectors: trajectory of 5 atoms, 10 steps,
    # "4d-coordinates"
    c_X = rand(20,4,10)
    X = rand(4,4)*5
    Y = rand(4,4)*3
    c_Y = coord_trans(c_X, old=X, new=Y, axis=1)
    c_X2 = coord_trans(c_Y, old=Y, new=X, axis=1)
    asrt_almost_equal(c_X, c_X2)
    
    #-----------------------------------------------------------
    # ND
    #-----------------------------------------------------------
    
    # arbitrary collection of 4-vectors
    c_X = rand(20,4,10,8)
    X = rand(4,4)*5
    Y = rand(4,4)*3
    c_Y = coord_trans(c_X, old=X, new=Y, axis=1)
    c_X2 = coord_trans(c_Y, old=Y, new=X, axis=1)
    asrt_almost_equal(c_X, c_X2)

