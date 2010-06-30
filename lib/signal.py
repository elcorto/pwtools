# signal.py
#
# Some general FFT/DFT stuff. Mostly textbook and reference implementations and
# utilities. FFT helper functions.
#

import numpy as np

def fftsample(a, b, mode='f', mirr=False):
    """Convert size and resolution between frequency and time domain.
    
    Convert between maximal frequency to sample (fmax) + desired frequency
    resolution (df) and the needed number of sample points (N) + time
    step (dt).
    
    The maximal frequency is also called the Nyquist frequency and is
    1/2*samplerate.
    
    args:
    -----
    a, b: see below
    mode : string, {'f', 't'}
        f : frequency mode
        t : time mode
    mirr: bool, consider mirroring of the signal at t=0 before Fourier
        transform

    f-mode:
        a : fmax  
        b : df 
    t-mode:
        a : dt 
        b : N

    returns:
    --------
    array([x,y])
    f-mode:
        x: dt  
        y: N
    t-mode:
        x: fmax 
        y: df
    
    notes:
    ------
    These relations hold ("v" - down, "^" - up):
        size                resolution
        N [t] ^     <->     df [f] v
        fmax [f] ^  <->     dt [t] v
    
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
    
    args:
    -----
    a : numpy 1d array
    method : string, {'matmul', 'loop'}
    
    returns: 
    --------
    (len(a),) array

    examples:
    ---------
    >>> from scipy.fftpack import fft
    >>> a=np.random.rand(100)
    >>> sfft=fft(a)
    >>> dfft1=dft(a, method='loop')
    >>> dfft2=dft(a, method='matmul')
    >>> np.testing.assert_array_almost_equal(sfft, dfft1)
    >>> np.testing.assert_array_almost_equal(sfft, dfft2)

    notes:
    ------
    This is only a reference implementation and has it's limitations.
        'loop': runs looong
        'matmul': memory limit
        => use only with medium size arrays

    N = len(a)
    sqrt(-1) == np.sqrt(1.0 + 0.0*j) = 1.0j

    Forward DFT, see [2,3], scipy.fftpack.fft():
        y[k] = sum(n=0...N-1) a[n] * exp(-2*pi*n*k*sqrt(-1)/N)
        k = 0 ... N-1
    
    Backward DFT, see [1] eq. 12.1.6, 12.2.2:
        y[k] = sum(n=0...N-1) a[n] * exp(2*pi*n*k*sqrt(-1)/N)
        k = 0 ... N-1

    The algo for method=='matmul' is the matrix mult from [1], but as Forward
    DFT for comparison with scipy. The difference between FW and BW DFT is that
    the imaginary parts are mirrored around y=0. 

    [1] Numerical Recipes in Fortran, Second Edition, 1992
    [2] http://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
    [3] http://mathworld.wolfram.com/FourierTransform.html
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
    
    args:
    -----
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
    
    returns:
    --------
    padded array

    examples:
    ---------
    # 1d 
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
    # 2d
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
    # up to next power of two           
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

