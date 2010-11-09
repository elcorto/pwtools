from math import sqrt
import numpy as np
from pwtools.crys import coord_trans

def test():
    c_X = np.random.rand(20,3)
    # basis vecs are assumed to be rows
    X = np.random.rand(3,3)*5
    Y = np.random.rand(3,3)*3

    # transform and back-transform
    c_Y = coord_trans(c_X, old=X, new=Y, align='rows')
    c_X2 = coord_trans(c_Y, old=Y, new=X, align='rows')
    np.testing.assert_array_almost_equal(c_X, c_X2)

    # with basis vecs aligned as cols
    c_Y2 = coord_trans(c_X, old=X.T, new=Y.T, align='cols')
    np.testing.assert_array_almost_equal(c_Y, c_Y2)

    # `cell` must have the right shape
    try:
        coord_trans(np.random.rand(20,4), old=X, new=Y, align='rows')
    except AssertionError:
        print "KNOWNFAIL"

    # simple dot product must produce same cartesian results:
    # v_X . X = v_I . I = v_I
    X = np.identity(3)
    Y = np.random.rand(3,3)*3
    c_X = np.random.rand(20,3)
    c_Y = coord_trans(c_X, old=X, new=Y, align='rows')
    # normal back-transform
    c_X2 = coord_trans(c_Y, old=Y, new=X, align='rows')
    # 2 forms w/ dot(), assume: basis vecs = rows of X and Y
    c_X3 = np.dot(c_Y, Y)
    c_X4 = np.dot(Y.T, c_Y.T).T
    np.testing.assert_array_almost_equal(c_X, c_X2)
    np.testing.assert_array_almost_equal(c_X, c_X3)
    np.testing.assert_array_almost_equal(c_X, c_X4)

    # some textbook example
    #
    v_I = np.array([1.0,1.5])
    I = np.identity(2)
    X = sqrt(2)/2.0*np.array([[1,-1],[1,1]])
    Y = np.array([[1,1],[0,1]])

    # "identity" transform
    np.testing.assert_array_almost_equal(coord_trans(v_I,I,I,align='cols'), v_I)

    # v in basis X and Y
    v_X = coord_trans(v_I,I,X,align='cols')
    v_Y = coord_trans(v_I,I,Y,align='cols')
    # We don't test v_Y b/c the numbers are not as pretty as [-0.5, 1.5]. Maybe
    # choose better example.
    np.testing.assert_array_almost_equal(v_Y, np.array([-0.5,  1.5]))

    # back-transform
    np.testing.assert_array_almost_equal(coord_trans(v_X,X,I,align='cols'), v_I)
    np.testing.assert_array_almost_equal(coord_trans(v_Y,Y,I,align='cols'), v_I)
