import numpy as np
from pwtools import mpl
from pwtools.test import tools
rand = np.random.rand

def assert_equal(d1, d2, keys=None):
    if keys is None:
        keys = ['x','y','xx','yy','zz','X','Y','Z','XY']
    tools.assert_dict_with_all_types_equal(d1.__dict__, d2.__dict__,
                                           keys=keys)

def test_data2d():
    d1 = mpl.get_2d_testdata()
    for k in [d1.x, d1.y, d1.xx, d1.yy, d1.zz, d1.X, d1.Y, d1.Z, d1.XY]:
        assert k is not None
    
    # test various forms of x-y input
    d2 = mpl.Data2D(x=d1.x, y=d1.y, zz=d1.zz)
    assert_equal(d1, d2)    
    d2 = mpl.Data2D(x=d1.x, y=d1.y, Z=d1.Z)
    assert_equal(d1, d2)    
    d2 = mpl.Data2D(xx=d1.xx, yy=d1.yy, Z=d1.Z)
    assert_equal(d1, d2)      
    d2 = mpl.Data2D(X=d1.X, Y=d1.Y, Z=d1.Z)
    assert_equal(d1, d2)      
    d2 = mpl.Data2D(XY=d1.XY, Z=d1.Z)
    assert_equal(d1, d2)      

    # Z data is optional
    d2 = mpl.Data2D(XY=d1.XY)
    assert_equal(d1, d2, keys=['x','y','xx','yy','X','Y','XY'])      


def test_data2d_unique_order():
    for x,y in [(np.array([1,2,3,4,5,6]), np.array([1,3,-3,-7,-8])),
                (rand(12), rand(222))]:

        d1 = mpl.Data2D(x=x, y=y)
        d2 = mpl.Data2D(xx=d1.xx, yy=d1.yy)
        
        # test Data2D._unique() 
        #
        # with np.unique()
        #   d1.y = array([1,  3,  -3, -7, -8])
        #   d2.y = np.unique(d1.yy)
        #        = array([-8, -7, -3,  1,  3])
        # instead of
        #   d2.y = Data2D._unique(d1.yy)
        #        = array([1,  3,  -3, -7, -8])
        #        = d1.y
        assert_equal(d1, d2)    
