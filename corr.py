# corr.py
#
# (Auto)correlation stuff. Reference implementations.
#

from scipy.fftpack import fft, ifft
import numpy as np
import _flib

def acorr(v, method=7):
    """Normalized autocorrelation function (ACF) for 1d arrys: 
    c(t) = <v(0) v(t)> / <v(0)**2>. 
    The x-axis is the offset "t" (or "lag" in Digital Signal Processing lit.).

    Several Python and Fortran implememtations. The Python versions are mostly
    for reference and are slow, except for fft-based, which is by far the
    fastet. 

    args:
    -----
    v : 1d array
    method : int
        1: Python loops
        2: Python loops, zero-padded
        3: method 1, numpy vectorized
        4: uses numpy.correlate()
        5: Fortran version of 1
        6: Fortran version of 3
        7: fft, Wiener-Khinchin Theorem
    
    returns:
    --------
    c : numpy 1d array
        c[0]  <=> lag = 0
        c[-1] <=> lag = len(v)
    
    notes:
    ------
    speed:
        methods 1 ...  are loosely ordered slow ... fast
    methods:
       All methods, besides the FFT, are "exact", they use variations of loops
       in the time domain, i.e. norm(acorr(v,1) - acorr(v,6)) = 0.0. 
       The FFT method introduces small numerical noise, norm(acorr(v,1) -
       acorr(v,4)) = O(1e-16) or so.

    signature of the Fortran extension _flib.acorr
        acorr - Function signature:
          c = acorr(v,c,method,[nstep])
        Required arguments:
          v : input rank-1 array('d') with bounds (nstep)
          c : input rank-1 array('d') with bounds (nstep)
          method : input int
        Optional arguments:
          nstep := len(v) input int
        Return objects:
          c : rank-1 array('d') with bounds (nstep)
    
    refs:
    -----
    [1] Numerical Recipes in Fortran, 2nd ed., ch. 13.2
    [2] http://mathworld.wolfram.com/FourierTransform.html
    [3] http://mathworld.wolfram.com/Cross-CorrelationTheorem.html
    [4] http://mathworld.wolfram.com/Wiener-KhinchinTheorem.html
    [5] http://mathworld.wolfram.com/Autocorrelation.html
    """
    nstep = v.shape[0]
    c = np.zeros((nstep,), dtype=float)
    if method == 1:
        for t in xrange(nstep):    
            for j in xrange(nstep-t):
                c[t] += v[j]*v[j+t] 
    elif method == 2:
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        for t in xrange(nstep):    
            for j in xrange(nstep):
                c[t] += v[j]*vv[j+t] 
    elif method == 3: 
        for t in xrange(nstep):
            c[t] = (v[:(nstep-t)] * v[t:]).sum()
    elif method == 4: 
        # old_behavior : for numpy 1.4.x
        c = np.correlate(v, v, mode='full', old_behavior=False)[nstep-1:]
    elif method == 5: 
        return _flib.acorr(v, c, 1)
    elif method == 6: 
        return _flib.acorr(v, c, 2)
    elif method == 7: 
        # Correlation via fft. After ifft, the imaginary part is (in theory) =
        # 0, in practise < 1e-16.
        # Cross-Correlation Theorem:
        #   corr(a,b)(t) = Int(-oo, +oo) a(tau)*conj(b)(tau+t) dtau   
        #                = ifft(fft(a)*fft(b).conj())
        # If a == b (like here), then this reduces to the special case of the 
        # Wiener-Khinchin Theorem (autocorrelation of `a`):
        #   corr(a,a) = ifft(np.abs(fft(a))**2)
        # Both theorems assume *periodic* data, i.e. `a` and `b` repeat after
        # `nstep` points. To deal with non-periodic data, we use zero-padding
        # at the end of `a` [1]. The result `c` contains the correlations for
        # positive and negative lags. Since the ACF is symmetric around
        # lag=0, we return 0 ... +lag.
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        c = ifft(np.abs(fft(vv))**2.0)[:nstep].real
    else:
        raise ValueError('unknown method: %s' %method)
    return c / c[0]

