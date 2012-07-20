# signal.py
#
# After scipy.signal: Some general "signal procressing" tools (FFT,
# correlation). Mostly textbook and reference implementations and utilities.

import numpy as np
from scipy.fftpack import fft, ifft
from pwtools import _flib

def fftsample(a, b, mode='f', mirr=False):
    """Convert size and resolution between frequency and time domain.
    
    Convert between maximal frequency to sample (fmax) + desired frequency
    resolution (df) and the needed number of sample points (N) + time
    step (dt).
    
    The maximal frequency is also called the Nyquist frequency and is
    1/2*samplerate.
    
    Parameters
    ----------
    a, b: float
        | mode='f': a=fmax  b=df
        | mode='t': a=dt    b=N
    mode : string, {'f', 't'}
        | f : frequency mode
        | t : time mode
    mirr: bool 
        consider mirroring of the signal at t=0 before Fourier transform
    
    Returns
    -------
    mode='f': array([dt,   N])
    mode='t': array([fmax, df])
    
    Examples
    --------
    >>> # fmax = 100 Hz, df = 1 Hz -> you need 200 steps with dt=0.005 sec
    >>> fftsample(100, 1, mode='f') 
    array([  5.00000000e-03,   2.00000000e+03])
    >>> fftsample(5e-3, 2e3, mode='t')
    array([ 100. ,    1.])
    # If you mirror, you only need 100 steps
    >>> fftsample(100, 1, mode='f', mirr=True)
    array([  5.00000000e-03,   1.00000000e+02])

    Notes
    -----
    These relations hold:
    
    ===========         ===========
    size                resolution
    ===========         ===========
    N [t] up            df [f] down
    fmax [f] up         dt [t] down
    ===========         ===========
    
    If you know that the signal in the time domain will be mirrored before FFT
    (N -> 2*N), you will get 1/2*df (double fine resolution), so 1/2*N is
    sufficient to get the desired df.

    Units:
    In general frequency_unit = 1/time_unit, need not be Hz and s.
    """
    if mode == 'f':
        fmax, df = a,b
        if mirr:
            df *= 2
        dt = 0.5/fmax
        N = 1.0/(df*dt)
        return np.array([dt, N])
    elif mode == 't':
        dt, N = a, b
        if mirr:
            N *= 2
        fmax = 0.5/dt
        df = 1.0/(N*dt)
        return np.array([fmax, df])
    else:
        raise ValueError("illegal mode, allowed: t, f")


def dft(a, method='loop'):
    """Simple straightforward complex DFT algo. 
    
    Parameters
    ----------
    a : numpy 1d array
    method : string, {'matmul', 'loop'}
    
    Returns
    -------
    (len(a),) array

    Examples
    --------
    >>> from scipy.fftpack import fft
    >>> a=np.random.rand(100)
    >>> sfft=fft(a)
    >>> dfft1=dft(a, method='loop')
    >>> dfft2=dft(a, method='matmul')
    >>> np.testing.assert_array_almost_equal(sfft, dfft1)
    >>> np.testing.assert_array_almost_equal(sfft, dfft2)

    Notes
    -----
    This is only a reference implementation and has it's limitations.
        | 'loop': runs looong
        | 'matmul': memory limit
        | => use only with medium size arrays

    N = len(a)
    sqrt(-1) == np.sqrt(1.0 + 0.0*j) = 1.0j

    Forward DFT, see [2]_ and [3]_ , scipy.fftpack.fft():
        y[k] = sum(n=0...N-1) a[n] * exp(-2*pi*n*k*sqrt(-1)/N)
        k = 0 ... N-1
    
    Backward DFT, see [1]_ eq. 12.1.6, 12.2.2:
        y[k] = sum(n=0...N-1) a[n] * exp(2*pi*n*k*sqrt(-1)/N)
        k = 0 ... N-1

    The algo for method=='matmul' is the matrix mult from [1]_, but as Forward
    DFT for comparison with scipy. The difference between FW and BW DFT is that
    the imaginary parts are mirrored at y=0. 
    
    References
    ----------
    .. [1] Numerical Recipes in Fortran, Second Edition, 1992
    .. [2] http://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
    .. [3] http://mathworld.wolfram.com/FourierTransform.html
    """
    pi = np.pi
    N = a.shape[0]
    # n and k run from 0 ... N-1
    nk = np.linspace(0.0, float(N), endpoint=False, num=N)

    if method == 'loop':
        fta = np.empty((N,), dtype=complex)
        for k in nk:
            fta[k] = np.sum(a*np.exp(-2.0*pi*1.0j*k*nk/float(N)))
    elif method == 'matmul':
        # `mat` is the matrix with elements W**(n*k) in [1], eq. 12.2.2
        nkmat = nk*nk[:,np.newaxis]
        mat = np.exp(-2.0*pi*1.0j*nkmat/float(N))
        fta = np.dot(mat, a)
    else:
        raise ValueError("illegal method '%s'" %method)
    return fta            


def dft_axis(arr, axis=-1):
    """Same as scipy.fftpack.fft(arr, axis=axis), but *much* slower."""
    return np.apply_along_axis(dft, axis, arr)
 

def pad_zeros(arr, axis=0, where='end', nadd=None, upto=None, tonext=None,
              tonext_min=None):
    """Pad an nd-array with zeros. Default is to append an array of zeros of 
    the same shape as `arr` to arr's end along `axis`.
    
    Parameters
    ----------
    arr :  nd array
    axis : the axis along which to pad
    where : string {'end', 'start'}, pad at the end ("append to array") or 
        start ("prepend to array") of `axis`
    nadd : number of items to padd (i.e. nadd=3 means padd w/ 3 zeros in case
        of an 1d array)
    upto : pad until arr.shape[axis] == upto
    tonext : bool, pad up to the next power of two (pad so that the padded 
        array has a length of power of two)
    tonext_min : int, when using `tonext`, pad the array to the next possible
        power of two for which the resulting array length along `axis` is at
        least `tonext_min`; the default is tonext_min = arr.shape[axis]

    Use only one of nadd, upto, tonext.
    
    Returns
    -------
    padded array

    Examples
    --------
    >>> # 1d 
    >>> pad_zeros(a)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=3)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, upto=6)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=1)
    array([1, 2, 3, 0])
    >>> pad_zeros(a, nadd=1, where='start')
    array([0, 1, 2, 3])
    >>> # 2d
    >>> a=arange(9).reshape(3,3)
    >>> pad_zeros(a, nadd=1, axis=0)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8],
           [0, 0, 0]])
    >>> pad_zeros(a, nadd=1, axis=1)
    array([[0, 1, 2, 0],
           [3, 4, 5, 0],
           [6, 7, 8, 0]])
    >>> # up to next power of two           
    >>> 2**arange(10)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256, 512])
    >>> pydos.pad_zeros(arange(9), tonext=True).shape
    (16,)
    """
    if tonext == False:
        tonext = None
    lst = [nadd, upto, tonext]
    assert lst.count(None) in [2,3], "`nadd`, `upto` and `tonext` must be " +\
           "all None or only one of them not None"
    if nadd is None:
        if upto is None:
            if (tonext is None) or (not tonext):
                # default
                nadd = arr.shape[axis]
            else:
                tonext_min = arr.shape[axis] if (tonext_min is None) \
                             else tonext_min
                # beware of int overflows starting w/ 2**arange(64), but we
                # will never have such long arrays anyway
                two_powers = 2**np.arange(30)
                if tonext_min > two_powers[-1]:
                    print "[pad_zeros]: WARNING: required array length longer \
                           than highest power of two, will not pad"
                    nadd = 0
                else:
                    power = two_powers[np.searchsorted(two_powers,
                                                      tonext_min)]
                    nadd = power - arr.shape[axis]                                                       
        else:
            nadd = upto - arr.shape[axis]
    if nadd == 0:
        return arr
    add_shape = list(arr.shape)
    add_shape[axis] = nadd
    add_shape = tuple(add_shape)
    if where == 'end':
        return np.concatenate((arr, np.zeros(add_shape, dtype=arr.dtype)), axis=axis)
    elif where == 'start':        
        return np.concatenate((np.zeros(add_shape, dtype=arr.dtype), arr), axis=axis)
    else:
        raise StandardError("illegal `where` arg: %s" %where)


def welch(M, sym=1):
    """Welch window. Function skeleton shamelessly stolen from
    scipy.signal.bartlett() and others."""
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1,dtype=float)
    odd = M % 2
    if not sym and not odd:
        M = M+1
    n = np.arange(0,M)
    w = 1.0-((n-0.5*(M-1))/(0.5*(M-1)))**2.0
    if not sym and not odd:
        w = w[:-1]
    return w

# Generalization to correlation corr(v,w) should be straightforward.
# Autocorrelation is then corr(v,v).
def acorr(v, method=7, norm=True):
    """(Normalized) autocorrelation function (ACF) for 1d arrays:
    Without normalization
        c(t) = <v(0) v(t)>
    and with
        c(t) = <v(0) v(t)> / <v(0)**2>
            
    The x-axis is the offset "t" (or "lag" in Digital Signal Processing lit.).

    Several Python and Fortran implememtations. The Python versions are mostly
    for reference and are slow, except for fft-based, which is by far the
    fastet. 

    Parameters
    ----------
    v : 1d array
    method : int
        | 1: Python loops
        | 2: Python loops, zero-padded
        | 3: method 1, numpy vectorized
        | 4: uses numpy.correlate()
        | 5: Fortran version of 1
        | 6: Fortran version of 3
        | 7: fft, Wiener-Khinchin Theorem
    norm : bool
        normalize or not

    Returns
    -------
    c : numpy 1d array
        | c[0]  <=> lag = 0
        | c[-1] <=> lag = len(v)
    
    Notes
    -----
    speed:
        methods 1 ...  are loosely ordered slow ... fast
    methods:
       All methods, besides the FFT, are "exact", they use variations of loops
       in the time domain, i.e. norm(acorr(v,1) - acorr(v,6)) = 0.0. 
       The FFT method introduces small numerical noise, norm(acorr(v,1) -
       acorr(v,4)) = O(1e-16) or so.

    signature of the Fortran extension _flib.acorr::

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
    
    References
    ----------
    .. [1] Numerical Recipes in Fortran, 2nd ed., ch. 13.2
    .. [2] http://mathworld.wolfram.com/FourierTransform.html
    .. [3] http://mathworld.wolfram.com/Cross-CorrelationTheorem.html
    .. [4] http://mathworld.wolfram.com/Wiener-KhinchinTheorem.html
    .. [5] http://mathworld.wolfram.com/Autocorrelation.html
    """
    nstep = v.shape[0]
    c = np.zeros((nstep,), dtype=float)
    _norm = 1 if norm else 0
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
        return _flib.acorr(v, c, 1, _norm)
    elif method == 6: 
        return _flib.acorr(v, c, 2, _norm)
    elif method == 7: 
        # Correlation via fft. After ifft, the imaginary part is (in theory) =
        # 0, in practise < 1e-16.
        # Cross-Correlation Theorem:
        #   corr(a,b)(t) = Int(-oo, +oo) a(tau)*conj(b)(tau+t) dtau   
        #                = ifft(fft(a)*fft(b).conj())
        # If a == b (like here), then this reduces to the special case of the 
        # Wiener-Khinchin Theorem (autocorrelation of `a`):
        #   corr(a,a) = ifft(np.abs(fft(a))**2)
        # Note that fft(a) is complex in gereal and abs() must be used!  Both
        # theorems assume *periodic* data, i.e. `a` and `b` repeat after
        # `nstep` points. To deal with non-periodic data, we use zero-padding
        # at the end of `a` [1]. The result `c` contains the correlations for
        # positive and negative lags. Since the ACF is symmetric around lag=0,
        # we return 0 ... +lag.
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        c = ifft(np.abs(fft(vv))**2.0)[:nstep].real
    else:
        raise ValueError('unknown method: %s' %method)
    if norm:        
        return c / c[0]
    else:
        return c

