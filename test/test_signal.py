import numpy as np
from scipy.signal import hanning, gaussian
from pwtools.signal import gauss, find_peaks, smooth, fft_1d_loop
from scipy.fftpack import fft
from pwtools import num
rand = np.random.rand

def test_slices():
    # Test if the fancy slice stunts in smooth() do what we want, i.e. test
    # if numpy has changed.
    y = np.arange(7) 
    npad = len(y)
    ysl = y[1:][::-1]
    assert (ysl == y[slice(npad,0,-1)]).all()
    assert (ysl == np.array([6, 5, 4, 3, 2, 1])).all()
    ysl = y[:-1][::-1]
    assert (ysl == y[slice(-2,-(npad+2),-1)]).all()
    assert (ysl == np.array([5, 4, 3, 2, 1, 0])).all()


def test_smooth_1d():
    for edge in ['m', 'c']:
        for N in [20,21]:
            # values in [9.0,11.0]
            x = rand(N) + 10
            mn = 9.0
            mx = 11.0
            for M in range(18,27):
                print "1d", edge, "N=%i, M=%i" %(N,M)
                xsm = smooth(x, gaussian(M,2.0), edge=edge)
                assert len(xsm) == N
                # (N,1) case
                xsm2 = smooth(x[:,None], gaussian(M,2.0)[:,None], edge=edge)
                assert np.allclose(xsm, xsm2[:,0], atol=1e-14, rtol=1e-12)
                # Smoothed signal should not go to zero if edge effects are handled
                # properly. Also assert proper normalization (i.e. smoothed signal
                # is "in the middle" of the noisy original data).
                assert xsm.min() >= mn
                assert xsm.max() <= mx
                assert mn <= xsm[0] <= mx
                assert mn <= xsm[-1] <= mx
            # convolution with delta peak produces same data exactly
            assert np.allclose(smooth(x, np.array([0.0,1,0]), edge=edge),x, atol=1e-14,
                               rtol=1e-12)


def test_smooth_nd():
    for edge in ['m', 'c']:
        a = rand(20, 2, 3) + 10
        for M in [5, 20, 123]:
            print "nd", edge, "M=%i" %M
            kern = gaussian(M, 2.0)
            asm = smooth(a, kern[:,None,None], axis=0, edge=edge)
            assert asm.shape == a.shape
            for jj in range(asm.shape[1]):
                for kk in range(asm.shape[2]):
                    assert np.allclose(asm[:,jj,kk], smooth(a[:,jj,kk], kern, 
                                                            edge=edge))
                    mn = a[:,jj,kk].min()
                    mx = a[:,jj,kk].max()
                    smn = asm[:,jj,kk].min()
                    smx = asm[:,jj,kk].max()
                    assert smn >= mn, "min: data=%f, smooth=%f" %(mn, smn)
                    assert smx <= mx, "max: data=%f, smooth=%f" %(mx, smx)


def test_find_peaks():
    x = np.linspace(0,10,300) 
    y = 0.2*gauss(x-0.5,.1) + gauss(x-2,.1) + 0.7*gauss(x-3,0.1) + gauss(x-6,1)
    # ymin=0.4: ignore first peak at x=0.5
    idx0, pos0 = find_peaks(y,x, ymin=0.4)
    assert idx0 == [60, 90, 179] 
    assert np.allclose(pos0, np.array([2,3,6.]), atol=1e-3)


def test_fft_1d_loop():
    a = rand(10,20,30,40)
    for axis in [0,1,2,3]:
        assert (fft(a, axis=axis) == fft_1d_loop(a, axis=axis)).all()
    a = rand(10)
    assert (fft(a) == fft_1d_loop(a)).all()

