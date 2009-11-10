# fft.py
#
# Some general FFT/DFT stuff. Mostly textbook and reference implementations and
# utilities.
#

import numpy as np

def fftsample(a, b, mode='f', mirr=False):
    """Convert size and resolution between time and frequency domain.
    
    Convert between maximal frequency to sample + desired frequency
    resolution and the needed number of sample points and the time step.
    
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
        a : fmax, max. freq to sample [Hz] == Nyquist freq. == 1/2 samplerate
        b : df, desired freq. resolution [Hz]
    t-mode:
        a : dt
        b : N

    returns:
    --------
    array([x,y])
    f-mode:
        x: dt : time step, unit is [s] (or in general 1/unit_of_fmax)
        y: N : number of samples
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

#-----------------------------------------------------------------------------

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


#-----------------------------------------------------------------------------

def dft_axis(arr, axis=-1):
    """Same as scipy.fftpack.fft(arr, axis=axis), but *much* slower."""
    return np.apply_along_axis(dft, axis, arr)
 

