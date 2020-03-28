"""
Some general signal procressing tools (FFT, correlation). Mostly textbook and
reference implementations plus some utilities.
"""

from itertools import product
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve, gaussian, kaiserord, firwin, lfilter, freqz
from scipy.integrate import trapz
from pwtools import _flib, num


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
    sqrt(complex(-1)) = np.sqrt(-1 + 0*j) = 1j

    Forward DFT, see [2]_ and [3]_ , scipy.fftpack.fft():
        y[k] = sum(n=0...N-1) a[n] * exp(-2*pi*n*k*j/N)
        k = 0 ... N-1

    Backward DFT, see [1]_ eq. 12.1.6, 12.2.2:
        y[k] = sum(n=0...N-1) a[n] * exp(2*pi*n*k*j/N)
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
        for ik,k in enumerate(nk):
            fta[ik] = np.sum(a*np.exp(-2.0*pi*1.0j*k*nk/float(N)))
    elif method == 'matmul':
        # `mat` is the matrix with elements W**(n*k) in [1], eq. 12.2.2
        nkmat = nk*nk[:,np.newaxis]
        mat = np.exp(-2.0*pi*1.0j*nkmat/float(N))
        fta = np.dot(mat, a)
    else:
        raise ValueError("illegal method '%s'" %method)
    return fta


def ezfft(y, dt=1.0):
    """Simple FFT function for interactive use.

    Parameters
    ----------
    y : 1d array to fft
    dt : float
        time step

    Returns
    -------
    faxis, fft(y)

    Examples
    --------
    >>> t = linspace(0,1,200)
    >>> x = sin(2*pi*10*t) + sin(2*pi*20*t)
    >>> f,d = signal.ezfft(x, dt=t[1]-t[0])
    >>> plot(f,abs(d))
    """
    assert y.ndim == 1
    faxis = np.fft.fftfreq(len(y), dt)
    split_idx = len(faxis)/2
    return faxis[:split_idx], fft(y)[:split_idx]


def fft_1d_loop(arr, axis=-1):
    """Like scipy.fft.pack.fft and numpy.fft.fft, perform fft along an axis.
    Here do this by looping over remaining axes and perform 1D FFTs.

    This was implemented as a low-memory version like
    :func:`~pwtools.crys.smooth` to be used in :func:`~pwtools.pydos.pdos`,
    which fills up the memory for big MD data. But actually it has the same
    memory footprint as the plain scipy fft routine. Keep it here anyway as a
    nice reference for how to loop over remaining axes in the ndarray case.
    """
    if axis < 0:
        axis = arr.ndim - 1
    axes = [ax for ax in range(arr.ndim) if ax != axis]
    # tuple here is 3x faster than generator expression
    #   idxs = (range(arr.shape[ax]) for ax in axes)
    idxs = tuple(range(arr.shape[ax]) for ax in axes)
    out = np.empty(arr.shape, dtype=complex)
    for idx_tup in product(*idxs):
        sl = [slice(None)] * arr.ndim
        for idx,ax in zip(idx_tup, axes):
            sl[ax] = idx
        tsl = tuple(sl)
        out[tsl] = fft(arr[tsl])
    return out



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
                assert tonext_min <= two_powers[-1], ("tonext_min exceeds "
                    "max power of 2")
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
        raise Exception("illegal `where` arg: %s" %where)


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

def lorentz(M, std=1.0, sym=True):
    r"""Lorentz window (same as Cauchy function). Function skeleton stolen from
    scipy.signal.gaussian().

    The Lorentz function is

    .. math::

        L(x) = \frac{\Gamma}{(x-x_0)^2 + \Gamma^2}

    Here :math:`x_0 = 0` and `std` = :math:`\Gamma`.
    Some definitions use :math:`1/2\,\Gamma` instead of :math:`\Gamma`, but
    without 1/2 we get comparable peak width to Gaussians when using this
    window in convolutions, thus ``scipy.signal.gaussian(M, std=5)`` is similar
    to ``lorentz(M, std=5)``.

    Parameters
    ----------
    M : int
        number of points
    std : float
        spread parameter :math:`\Gamma`
    sym : bool

    Returns
    -------
    w : (M,)
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1,dtype=float)
    odd = M % 2
    if not sym and not odd:
        M = M+1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    w = std / (n**2.0 + std**2.0)
    w /= w.max()
    if not sym and not odd:
        w = w[:-1]
    return w

cauchy = lorentz


def mirror(arr, axis=0):
    """Mirror array `arr` at index 0 along `axis`.
    The length of the returned array is 2*arr.shape[axis]-1 ."""
    return np.concatenate((arr[::-1],arr[1:]), axis=axis)

# XXX
# Check f2py wrapper of _flib.acorr(): The signature is:
#       In [13]: num._flib.acorr?
#       Type:           fortran
#       String form:    <fortran object>
#       Docstring:
#       c = acorr(v,c,method,norm,[nstep])
#
#       Wrapper for ``acorr``.
#
#       Parameters
#       ----------
#       v : input rank-1 array('d') with bounds (nstep)
#       c : input rank-1 array('d') with bounds (nstep)
#       method : input int
#       norm : input int
#
#       Other Parameters
#       ----------------
#       nstep : input int, optional
#           Default: len(v)
#
#       Returns
#       -------
#       c : rank-1 array('d') with bounds (nstep)
#
# We need to pass in a result array 'c' which gets overwritten, but this also
# gets returned. Check f2py docs for wrapping such that c generated on the
# Fortran side.
#
def acorr(v, method=7, norm=True):
    """(Normalized) autocorrelation function (ACF) for 1d arrays.

    Without normalization
        c(t) = <v(0) v(t)>
    and with
        c(t) = <v(0) v(t)> / <v(0)**2>

    The x-axis is the offset "t" (or "lag" in Digital Signal Processing lit.).
    Since the ACF is symmetric around t=0, we return only t=0...len(v)-1 .

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
    Generalization of this function to correlation corr(v,w) should be
    straightforward. Autocorrelation is then corr(v,v).

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
        for t in range(nstep):
            for j in range(nstep-t):
                c[t] += v[j]*v[j+t]
    elif method == 2:
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        for t in range(nstep):
            for j in range(nstep):
                c[t] += v[j]*vv[j+t]
    elif method == 3:
        for t in range(nstep):
            c[t] = (v[:(nstep-t)] * v[t:]).sum()
    elif method == 4:
        c = np.correlate(v, v, mode='full')[nstep-1:]
    elif method == 5:
        return _flib.acorr(v, c, 1, _norm)
    elif method == 6:
        return _flib.acorr(v, c, 2, _norm)
    elif method == 7:
        # Correlation via fft. After ifft, the imaginary part is (in theory) =
        # 0, in practise < 1e-16, so we are safe to return the real part only.
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        c = ifft(np.abs(fft(vv))**2.0)[:nstep].real
    else:
        raise ValueError('unknown method: %s' %method)
    if norm:
        return c / c[0]
    else:
        return c


def gauss(x, std=1.0, norm=False):
    """Gaussian function.

    Parameters
    ----------
    x : 1d array
    std : float
        sigma
    norm : bool
        Norm such that integrate(gauss(x),x=-inf,inf) = 1, i.e. normalize and
        return a PDF.

    Returns
    -------
    array_like(x)
    """
    if norm:
        return 1.0 / std / np.sqrt(2*np.pi) * np.exp(-x**2.0 / 2.0 / std**2.0)
    else:
        return np.exp(-x**2.0 / 2.0 / std**2.0)


def find_peaks(y, x=None, k=3, spread=2, ymin=None):
    """Simple peak finding algorithm.

    Find all peaks where ``y > ymin``. If `x` given, also extract peak maxima
    positions by fitting a spline of order `k` to each found peak. To find
    minima, just use ``-y``.

    Parameters
    ----------
    y : 1d array_like
        data with peaks
    x : 1d array_like, optional, len(y)
        x axis
    k : int
        order of spline
    spread : int
        Use ``2*spread+1`` points around each peak to fit a spline. Note that
        we need ``2*spread+1 > k``.
    ymin : float, optional
        Find all peaks above that value.

    Returns
    -------
    idx0, pos0
    idx0 : indices of peaks from finite diffs, each peak is at ``x[idx0[i]]``
    pos0 : refined `x`-positions of peaks if `x` given, else None

    Examples
    --------
    >>> from pwtools.signal import gauss, find_peaks
    >>> from pwtools import num
    >>> x=linspace(0,10,300); y=0.2*gauss(x-0.5,.1) + gauss(x-2,.1) + 0.7*gauss(x-3,0.1) + gauss(x-6,1)
    >>> # ymin=0.4: ignore first peak at x=0.5
    >>> find_peaks(y,x, ymin=0.4)
    ([60, 90, 179], [2.000231296097065, 3.0007122565950572, 5.999998055132549])
    >>> idx0, pos0=find_peaks(y,x, ymin=0.4)
    >>> spl=num.Spline(x,y)
    >>> plot(x,y)
    >>> for x0 in pos0:
    ...    plot([x0], [spl(x0)], 'ro')
    """
    ymin = y.min() if ymin is None else ymin
    idx0 = []
    dfy = np.diff(y, n=1)
    for ii in range(len(dfy)-1):
        if dfy[ii] > 0 and dfy[ii+1] < 0 and y[ii] >= ymin:
            idx0.append(ii+1)
    pos0 = None
    if x is not None:
        pos0 = []
        for i0 in idx0:
            sl = slice(i0-spread,i0+1+spread,None)
            xx = x[sl]
            yy = y[sl]
            spl = num.Spline(xx,-yy,k=k,s=None)
            try:
                root = spl.get_min()
                pos0.append(root)
            except ValueError:
                raise ValueError("error calculating spline maximum at idx=%i, x=%f" %(i0,x[i0]))
    return idx0, pos0


def smooth(data, kern, axis=0, edge='m', norm=True):
    """Smooth N-dim `data` by convolution with a kernel `kern`.

    Uses scipy.signal.fftconvolve().

    Note that due to edge effect handling (padding) and kernal normalization,
    the convolution identity convolve(data,kern) == convolve(kern,data) doesn't
    apply here. We always return an array of ``data.shape``.

    Parameters
    ----------
    data : nd array
        The data to smooth. Example: 1d (N,) or (N,K,3)
        for trajectory
    kern : nd array
        Convolution kernel. Example: 1d (M,) or (M,1,1)
        for trajectory along axis=0 (data length N)
    axis : int
        Axis along which to do the smoothing. That is actually not needed for
        the convolution ``fftconvolve(data, kern)`` but is used for padding the
        data along `axis` to handle edge effects before convolution.
    edge : str
        Method for edge effect handling.
            | 'm' : pad with mirror signal
            | 'c' : pad with constant values (i.e. ``data[0]`` and
            |       ``data[-1]`` in the 1d case)
    norm : bool
        Normalize kernel. Default is True. This assures that the smoothed
        signal lies within the data. Note that this is not True for kernels
        with very big spread (i.e. ``hann(N*10)`` or ``gaussian(N/2,
        std=N*10)``. Then the kernel is effectively a constant.

    Returns
    -------
    ret : data.shape
        Convolved signal.

    Examples
    --------
    >>> from pwtools.signal import welch
    >>> from numpy.random import rand
    >>> x = linspace(0,2*pi,500); a=cos(x)+rand(500)
    >>> plot(a, color='0.7')
    >>> k=scipy.signal.hann(21)
    >>> plot(signal.smooth(a,k), 'r', label='hann')
    >>> k=scipy.signal.gaussian(21, 3)
    >>> plot(signal.smooth(a,k), 'g', label='gauss')
    >>> k=welch(21)
    >>> plot(signal.smooth(a,k), 'y', label='welch')
    >>> legend()
    >>> # odd kernel [0,1,0] reproduces data exactly, i.e. convolution with
    >>> # delta peak
    >>> figure(); title('smooth with delta [0,1,0]')
    >>> x=linspace(0,2*pi,15); k=scipy.signal.hann(3)
    >>> plot(cos(x))
    >>> plot(signal.smooth(cos(x),k), 'r')
    >>> legend()
    >>> # edge effects with normal convolution
    >>> figure(); title('edge effects')
    >>> x=rand(20)+10; k=scipy.signal.hann(11);
    >>> plot(x); plot(signal.smooth(x,k),label="smooth");
    >>> plot(scipy.signal.convolve(x,k/k.sum(),'same'), label='convolve')
    >>> legend()
    >>> # edge effect methods
    >>> figure(); title('edge effect methods')
    >>> x=rand(20)+10; k=scipy.signal.hann(20);
    >>> plot(x); plot(signal.smooth(x,k,edge='m'),label="edge='m'");
    >>> plot(signal.smooth(x,k,edge='c'),label="edge='c'");
    >>> legend()
    >>> # smooth a trajectory of atomic coordinates
    >>> figure(); title('trajectory')
    >>> x = linspace(0,2*pi,500)
    >>> a = rand(500,2,3) # (nstep, natoms, 3)
    >>> a[:,0,:] += cos(x)[:,None]
    >>> a[:,1,:] += sin(x)[:,None]
    >>> k=scipy.signal.hann(21)[:,None,None]
    >>> y = signal.smooth(a,k)
    >>> plot(a[:,0,0], color='0.7'); plot(y[:,0,0],'b',
    ...                                   label='atom1 x')
    >>> plot(a[:,1,0], color='0.7'); plot(y[:,1,0],'r',
    ...                                   label='atom2 x')
    >>> legend()

    References
    ----------
    [1] http://wiki.scipy.org/Cookbook/SignalSmooth

    See Also
    --------
    :func:`welch`
    :func:`lorentz`

    Notes
    -----

    Kernels:

    Even kernels result in shifted signals, odd kernels are better.
    However, for N >> M, it doesn't make a difference really.

    Usual kernels (window functions) are created by e.g.
    ``scipy.signal.hann(M)``. For ``kern=scipy.signal.gaussian(M,
    std)``, two values are needed, namely `M` and `std`, where  `M`
    determines the number of points calculated for the convolution kernel, as
    in the other cases. But what is actually important is `std`, which
    determines the "used width" of the gaussian. Say we use N=100
    and M=50. That would be a massively wide window and we would
    smooth away all details. OTOH, using ``gaussian(50,3)`` would generate a
    kernel with the same number `M` of data points, but the gauss peak which is
    effectively used for convolution is much smaller. For ``gaussian()``,
    `M` should be bigger then `std`. The convolved signal will converge
    with increasing `M`. Good values are `M=6*std` and bigger. For
    :func:`lorentz`, much wider kernels are needed such as `M=100*std` b/c
    of the long tails of the Lorentz function. Testing is mandatory!

    Edge effects:

    We use padding of the signal with ``M=len(kern)`` values at both ends such
    that the convolution with `kern` doesn't zero the `data` at the signal
    edges. We have two methods. `edge='m'`: padd with the signal mirrored at 0
    and -1 or `edge='c'`: use the constant values ``data[0]`` and ``data[-1]``.
    Many more of these variants may be thought of. The choice of how to extend
    the data essentially involves an assumption about how the signal *would*
    continue, which is signal-dependent. In practice, we usually have ``M <<
    N`` (e.g. ``scipy.signal.hann(M)``) or ``std << N``
    (``scipy.signal.gaussian(M, std``). Then, both methods are identical in the
    middle and show only very small differences at the edges. Essentially, edge
    effect handling shall only ensure that the smoothed signal doesn't go to
    zero and that must be independent of the method, which is the case.

    Memory:

    For big data, fftconvolve() can easily eat up all your memory, for
    example::

    >>> # assume axis=0 is the axis along which to convolve
    >>> arr = ones((1e5,200,3))
    >>> kern = scipy.signal.hann(101)
    >>> ret = scipy.signal.fftconvolve(arr, kern[:,None,None])

    Then it is better to loop over some or all of the remaing dimensions::

    >>> ret = np.empty_like(arr)
    >>> for jj in range(arr.shape[1]):
    >>>     ret[:,jj,:] = smooth(arr[:,jj,:], kern[:,None])

    or::

    >>> for jj in range(arr.shape[1]):
    >>>     for kk in range(arr.shape[2]):
    >>>         ret[:,jj,kk] = smooth(arr[:,jj,kk], kern)

    The size of the chunk over which you explicitely loop depends on the data
    of course. We do exactly this in :func:`pwtools.crys.smooth`.
    """
    # edge = 'm'
    # ----------
    #
    # Add mirror of the signal left and right to handle edge effects, up to
    # signal length N on both ends. If M > N then fill padding regions up with
    # zeros until we have sig = [(M,), (N,), (M,)]. fftconvolve(..., 'valid')
    # always returns only the signal length where sig and kern overlap
    # completely. Therefore, data at the far end doesn't influence the edge and
    # we can safely put zeros (or anything else) there. The returned length is
    # always N+M+1.
    #
    # example (M < N), add M=3 data parts left and right
    # npad   = 3
    # data   =                 [1,2,3,4,5,6]
    # dleft  =           [4,3,2]
    # dright =                             [5,4,3]
    # sig    =           [4,3,2,1,2,3,4,5,6,5,4,3]
    # If M = 8 > N, then:
    # dleft  =       [6,5,4,3,2]
    # dright =                             [5,4,3,2,1]
    # sig    = [0,0,0,6,5,4,3,2,1,2,3,4,5,6,5,4,3,2,1,0,0,0]
    #
    # edge = 'c'
    # ----------
    # The same, but all padded values are the first (left) and last (right)
    # data value.
    N = data.shape[axis]
    M = kern.shape[axis]
    if edge == 'm':
        npad = min(M,N)
        sleft = slice(npad,0,-1)
        sright = slice(-2,-(npad+2),-1)
        dleft = num.slicetake(data, sl=sleft, axis=axis)
        dright = num.slicetake(data, sl=sright, axis=axis)
        assert dleft.shape == dright.shape
        K = dleft.shape[axis]
        if K < M:
            dleft  = pad_zeros(dleft,  axis=axis, where='start', nadd=M-K)
            dright = pad_zeros(dright, axis=axis, where='end',   nadd=M-K)
    elif edge == 'c':
        sl = [slice(None)]*data.ndim
        sl[axis] = None
        tsl = tuple(sl)
        dleft = np.repeat(num.slicetake(data, sl=0, axis=axis)[tsl], M, axis=axis)
        dright = np.repeat(num.slicetake(data, sl=-1, axis=axis)[tsl], M, axis=axis)
        assert dleft.shape == dright.shape
        # 1d special case: (M,1) -> (M,)
        if data.ndim == 1 and dleft.ndim == 2 and dleft.shape[1] == 1:
            dleft = dleft[:,0]
            dright = dright[:,0]
    else:
        raise Exception("unknown value for edge")
    sig = np.concatenate((dleft, data, dright), axis=axis)
    kk = kern/float(kern.sum()) if norm else kern
    ret = fftconvolve(sig, kk, 'valid')
    assert ret.shape[axis] == N+M+1, "unexpected convolve result shape"
    del sig
    if M % 2 == 0:
        ##sl = slice(M//2+1,-(M//2)) # even kernel, shift result to left
        sl = slice(M//2,-(M//2)-1) # even kernel, shift result to right
    else:
        sl = slice(M//2+1,-(M//2)-1)
    ret = num.slicetake(ret, sl=sl, axis=axis)
    assert ret.shape == data.shape, ("ups, ret.shape (%s)!= data.shape (%s)"
                                      %(ret.shape, data.shape))
    return ret


def odd(n, add=1):
    """Return next odd integer to `n`.

    Can be used to construt odd smoothing kernels in :func:`smooth`.

    Parameters
    ----------
    n : int
    add : int
        number to add if `n` is even, +1 or -1
    """
    assert add in [-1, 1], "add must be -1 or 1"
    return n if n % 2 == 1 else n + add


def scale(x, copy=True):
    """Scale `x` to unity.

    Subtract min and divide by (max-min).

    Parameters
    ----------
    x : array_like
    copy : bool
        copy `x` before scaling

    Returns
    -------
    x_scaled
    """
    xx = x.copy() if copy else x
    xx = xx - xx.min()
    xx /= xx.max()
    return xx


class FIRFilter(object):
    """Build and apply a digital FIR filter (low-, high-, band-pass,
    band-stop). Uses firwin() and in some cases kaiserord().

    Doc strings stolen from scipy.signal.

    Notes
    -----
    To plot the frequency response (the frequency bands), use::

        >>> f = Filter(...)
        >>> plot(f.w, abs(f.h))

    Examples
    --------
    .. literalinclude:: ../../../../examples/filter_example.py

    References
    ----------
    .. [1] http://www.scipy.org/Cookbook/FIRFilter
    """
    def __init__(self, cutoff, nyq, ntaps=None, ripple=None, width=None,
                 window='hamming', mode='lowpass'):
        """
        Parameters
        ----------
        cutoff : float or 1D array_like
            Cutoff frequency of filter (expressed in the same units as `nyq`)
            OR an array of cutoff frequencies (that is, band edges). In the
            latter case, the frequencies in `cutoff` should be positive and
            monotonically increasing between 0 and `nyq`.  The values 0 and
            `nyq` must not be included in `cutoff`.
        nyq : float
            Nyquist frequency [Hz].  Each frequency in `cutoff` must be between 0
            and `nyq`.
        ntaps : int
            Length of the filter (number of coefficients, i.e. the filter
            order + 1).  `ntaps` must be even if a passband includes the
            Nyquist frequency. Use either `ntaps` or `ripple` + `width` for a
            Kaiser window.
        ripple : float
            Positive number specifying maximum ripple in passband (dB) and
            minimum ripple in stopband. Large values (like 1000) remove the
            "rippling" in the pass band almost completely and make frequency
            response almost "square" (if `width` is small) but also
            lead to a large number of filter coeffs (ntaps).
        width : float
            Width of transition region (same unit as `nyq`, e.g. Hz).
        window : string or tuple of string and parameter values
            Desired window to use. See `scipy.signal.get_window` for a list
            of windows and required parameters. Default is "hamming". Ignored
            if `width` and `ripple` givem b/c then ``kaiserord`` is used to
            build a Kaiser window.
        mode : str
            'lowpass', 'highpass', 'bandpass', 'bandstop'
        """
        if ntaps is None:
            assert [ripple, width] != [None]*2, ("ntaps is None, we need "
                "ripple and width for a Kaiser window")
            self.ntaps, self.beta = kaiserord(float(ripple), float(width) / nyq)
            window = ('kaiser', self.beta)
        else:
            self.ntaps = ntaps
        self.window = window
        if mode == 'lowpass':
            pass_zero = True
        elif mode == 'highpass':
            pass_zero = False
        elif mode == 'bandpass':
            pass_zero = False
            assert len(cutoff) == 2
        elif mode == 'bandstop':
            pass_zero = True
            assert len(cutoff) == 2
            if N % 2 == 0:
                N += 1
        else:
            raise Exception('unknown mode')
        self.taps = firwin(numtaps=self.ntaps, cutoff=cutoff, window=self.window, nyq=nyq,
                           pass_zero=pass_zero, width=width)
        w,h = freqz(self.taps)
        self.w = (w/np.pi)*nyq
        self.h = h

    def __call__(self, x, axis=-1):
        """Apply filter to signal.

        Parameters
        ----------
        x : 1d array
        axis : int
        """
        return lfilter(self.taps, 1.0, x, axis=axis)
