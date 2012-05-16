import numpy as np
from pwtools import pwscf
from pwtools.test.tools import aae

def test_read_matdyn():
    # matdyn_freq
    kpoints, freqs = pwscf.read_matdyn_freq('files/matdyn.freq')
    aae(kpoints,
        np.array([[0.1, 0.2, 0.3],
                  [0.2, 0.4, 0.6]]))
    aae(freqs,
        np.array([[1,2,3,4,5,6.1,],
                  [7, 8, 9e-7,10, 11, 12]]))
    
    # matdyn_mods
    qpoints_ref = \
    np.array([[ 0. ,  0. ,  0. ],
              [ 0. ,  0. ,  0.5]])

    freqs_ref = \
    np.array([[  0.,   7.,  14.,  21.,  28.,  35.],
              [ 42.,  49.,  56.,  63.,  70.,  77.]])

    vecs_real_ref = \
    np.array([[[[  1.,   2.,   3.],
                [  4.,   5.,   6.]],
               [[  8.,   9.,  10.],
                [ 11.,  12.,  13.]],
               [[ 15.,  16.,  17.],
                [ 18.,  19.,  20.]],
               [[ 22.,  23.,  24.],
                [ 25.,  26.,  27.]],
               [[ 29.,  30.,  31.],
                [ 32.,  33.,  34.]],
               [[ 36.,  37.,  38.],
                [ 39.,  40.,  41.]]],
              [[[ 43.,  44.,  45.],
                [ 46.,  47.,  48.]],
               [[ 50.,  51.,  52.],
                [ 53.,  54.,  55.]],
               [[ 57.,  58.,  59.],
                [ 60.,  61.,  62.]],
               [[ 64.,  65.,  66.],
                [ 67.,  68.,  69.]],
               [[ 71.,  72.,  73.],
                [ 74.,  75.,  76.]],
               [[ 78.,  79.,  80.],
                [ 81.,  82.,  83.]]]])

    qpoints, freqs, vecs = pwscf.read_matdyn_modes('files/matdyn.modes',
                                                   natoms=2)
    aae(qpoints, qpoints_ref)
    aae(freqs, freqs_ref)
    aae(vecs.real, vecs_real_ref)
    assert not vecs.imag.any()
