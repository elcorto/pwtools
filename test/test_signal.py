import numpy as np
from scipy.signal import hanning
from pwtools.signal import gauss, find_peaks, smooth
from pwtools import num
rand = np.random.rand

def test_smooth_1d():
    for N in [20,21,99,100,101]:
        # values in [9.0,11.0]
        x = rand(N) + 10
        for M in [3,4,13,14]:
            xsm = smooth(x, hanning(M))
            assert len(xsm) == N
            # Smoothed signal should not go to zero if edge effects are handled
            # properly. Also assert proper normalization (i.e. smoothed signal
            # is "in the middle" of the noisy original data).
            assert xsm.min() >= 9.0 
            assert xsm.max() <= 11.0 
        # convolution with delta peak produces same data exactly
        assert np.allclose(smooth(x, np.array([0.0,1,0])),x)


def test_smooth_nd():
    # (500, 20, 3)
    a = rand(500)[:,None,None].repeat(20,1).repeat(3,2) + 10
    kern = hanning(21)
    asm = smooth(a, kern[:,None,None], axis=0)
    assert asm.shape == a.shape
    for jj in range(asm.shape[1]):
        for kk in range(asm.shape[2]):
            assert np.allclose(asm[:,jj,kk], smooth(a[:,jj,kk], kern))
            assert asm[:,jj,kk].min() >= 9.0 
            assert asm[:,jj,kk].max() <= 11.0 


def test_find_peaks():
    x = np.linspace(0,10,300) 
    y = 0.2*gauss(x-0.5,.1) + gauss(x-2,.1) + 0.7*gauss(x-3,0.1) + gauss(x-6,1)
    # ymin=0.4: ignore first peak at x=0.5
    idx0, pos0 = find_peaks(y,x, ymin=0.4)
    assert idx0 == [60, 90, 179] 
    assert np.allclose(pos0, np.array([2,3,6.]), atol=1e-3)

