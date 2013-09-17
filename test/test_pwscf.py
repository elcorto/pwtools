import numpy as np
from pwtools import pwscf

def test_atpos_str():
    assert pwscf.atpos_str(['X'], np.array([[0.1,1,1]]), fmt='%g', delim='_',
                           eps=0.0) == 'X_0.1_1_1'
    assert pwscf.atpos_str(['X'], np.array([[0.1,1,1]]), fmt='%g', delim='_',
                           eps=0.2) == 'X_0_1_1'                          
