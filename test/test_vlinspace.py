import numpy as np
from pwtools.num import vlinspace

def test():
    aa = np.array([0,0,0]) 
    bb = np.array([1,1,1])
    
    np.testing.assert_array_equal(vlinspace(aa,bb,1), aa[None,:])
    
    np.testing.assert_array_equal(vlinspace(aa,bb,2),
                                  np.array([aa,bb]))
    
    tgt = np.array([aa, [0.5,0.5,0.5], bb])
    np.testing.assert_array_equal(vlinspace(aa,bb,3), tgt)
    
    tgt = np.array([aa, [1/3.]*3, [2/3.]*3, bb])
    np.testing.assert_array_almost_equal(vlinspace(aa,bb,4), tgt)
    
    tgt = np.array([[ 0.  ,  0.  ,  0.  ],
                 [ 0.25,  0.25,  0.25],
                 [ 0.5 ,  0.5 ,  0.5 ],
                 [ 0.75,  0.75,  0.75]])
    np.testing.assert_array_equal(vlinspace(aa,bb,4,endpoint=False), 
                                  tgt)
    
    aa = np.array([-1,-1,-1]) 
    bb = np.array([1,1,1])
    tgt = np.array([[ -1.  ,  -1.  ,  -1.  ],
                    [ 0,  0,  0],
                    [ 1,  1,  1]])
    np.testing.assert_array_equal(vlinspace(aa,bb,3), 
                                  tgt)
                                        
