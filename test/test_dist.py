import numpy as np
from scipy.spatial.distance import cdist
from pwtools import num
from pwtools.test.tools import aaae
rand = np.random.rand

def test_dist():
    X = rand(100,5)
    Y = rand(80,5)
    d1 = num.distsq(X,Y)
    d2 = ((X[:,None,...] - Y[None,...])**2.0).sum(axis=-1)
    d3 = cdist(X,Y, metric='euclidean')**2.0
    print "d1 - d2"
    aaae(d1,d2)
    print "d1 - d3"
    aaae(d1,d3)
