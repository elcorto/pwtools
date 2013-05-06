import numpy as np
from pwtools.signal import gauss, find_peaks, smooth_convolve
from pwtools import num

def test_smooth_convolve():
    x = np.linspace(0,10,100) 
    y = np.random.rand(100)
    std = 0.2
    for area in [None,  np.trapz(y,x)]:
        ys_ref = smooth_convolve(x, y, std=std, area=area)
        for width in [None, 6*std, 12*std]:
            ys = smooth_convolve(x, y, std=std, width=width, area=area)
            print np.abs(ys_ref - ys).max()
            assert np.allclose(ys_ref, ys, rtol=1e-2)
    

def test_find_peaks():
    x = np.linspace(0,10,300) 
    y = 0.2*gauss(x-0.5,.1) + gauss(x-2,.1) + 0.7*gauss(x-3,0.1) + gauss(x-6,1)
    # ymin=0.4: ignore first peak at x=0.5
    idx0, pos0 = find_peaks(y,x, ymin=0.4)
    assert idx0 == [60, 90, 179] 
    assert np.allclose(pos0, np.array([2,3,6.]), atol=1e-3)

