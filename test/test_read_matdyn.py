import numpy as np
from pwtools.pwscf import read_matdyn_freq

def test():
    kpoints, freqs = read_matdyn_freq('files/matdyn.freq')
    np.testing.assert_array_equal(kpoints,
                                  np.array([[0.1, 0.2, 0.3],
                                            [0.2, 0.4, 0.6]]))
    np.testing.assert_array_equal(freqs,
                                  np.array([[1,2,3,4,5,6.1,],
                                            [7, 8, 9e-7,10, 11, 12]]))
