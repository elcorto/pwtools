# pydos.py
#
# This module implements the functionallity to calculate the phonon density of
# states (PDOS) from MD trajectories. For parsing output files into a format
# which is used here, see parse.py and test/* for examples. For a theory
# overview, see [1,2].
#
# [1] doc/source/written/background/phonon_dos.rst
# [2] http://elcorto.bitbucket.org/pwtools/written/background/phonon_dos.html
#
# Other codes wich do that:
# * tfreq from Tim Teatro
#   http://www.timteatro.net/2010/09/29/velocity-autocorrelation-and-vibrational-spectrum-calculation
# * fourier from CPMD

import os, warnings
import numpy as np
from scipy.fftpack import fft
from scipy.signal import convolve, gaussian
import constants, _flib, num
from pwtools.verbose import verbose
from pwtools.signal import pad_zeros, welch, mirror

def pyvacf(vel, m=None, method=3):
    """Reference implementation for calculating the VACF of velocities in 3d
    array `vel`. This is slow. Use for debugging only. For production, use
    fvacf().
    
    Parameters
    ----------
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        | 1 : 3 loops
        | 2 : replace 1 inner loop
        | 3 : replace 2 inner loops
    
    Returns
    -------
    c : 1d array (nstep,)
        VACF
    """
    natoms = vel.shape[1]
    nstep = vel.shape[0]
    c = np.zeros((nstep,), dtype=float)
    if m is None:
        m = np.ones((natoms,), dtype=float)
    if method == 1:
        # c(t) = <v(t0) v(t0 + t)> / <v(t0)**2> = C(t) / C(0)
        #
        # "displacements" `t'
        for t in range(nstep):
            # time origins t0 == j
            for j in range(nstep-t):
                for i in range(natoms):
                    c[t] += np.dot(vel[j,i,:], vel[j+t,i,:]) * m[i]
    elif method == 2:    
        # replace 1 inner loop
        for t in range(nstep):
            for j in range(nstep-t):
                # (natoms, 3) * (natoms, 1) -> (natoms, 3)
                c[t] += (vel[j,...] * vel[j+t,...] * m[:,None]).sum()
    elif method == 3:    
        # replace 2 inner loops:
        # (xx, natoms, 3) * (1, natoms, 1) -> (xx, natoms, 3)
        for t in range(nstep):
            c[t] = (vel[:(nstep-t),...] * vel[t:,...]*m[None,:,None]).sum()
    else:
        raise ValueError('unknown method: %s' %method)
    # normalize to unity
    c = c / c[0]
    return c


def fvacf(vel, m=None, method=2, nthreads=None):
    """Interface to Fortran function _flib.vacf(). Otherwise same
    functionallity as pyvacf(). Use this for production calculations.
    
    Parameters
    ----------
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        | 1 : loops
        | 2 : vectorized loops

    nthreads : int ot None
        If int, then use this many OpenMP threads in the Fortran extension.
        Only useful if the extension was compiled with OpenMP support, of
        course.

    Returns
    -------
    c : 1d array (nstep,)
        VACF

    Notes
    -----
    Fortran extension::

        $ python -c "import _flib; print _flib.vacf.__doc__"
        vacf - Function signature:
          c = vacf(v,m,c,method,use_m,[nthreads,natoms,nstep])
        Required arguments:
          v : input rank-3 array('d') with bounds (natoms,3,nstep)
          m : input rank-1 array('d') with bounds (natoms)
          c : input rank-1 array('d') with bounds (nstep)
          method : input int
          use_m : input int
        Optional arguments:
          nthreads : input int
          natoms := shape(v,0) input int
          nstep := shape(v,2) input int
        Return objects:
          c : rank-1 array('d') with bounds (nstep)
    
    Shape of `vel`: The old array shapes were (natoms, 3, nstep), the new is
        (nstep,natoms,3). B/c we don't want to adapt flib.f90, we change
        vel's shape before passing it to the extension.

    See Also
    --------
    :mod:`pwtools._flib`
    :func:`vacf_pdos`
    """
    # f2py copies and C-order vs. Fortran-order arrays
    # ------------------------------------------------
    # With vel = np.asarray(vel, order='F'), we convert vel to F-order and a
    # copy is made by numpy. If we don't do it, the f2py wrapper code does.
    # This copy is unavoidable, unless we allocate the array vel in F-order in
    # the first place.
    #   c = _flib.vacf(np.asarray(vel, order='F'), m, c, method, use_m)
    # 
    # speed
    # -----
    # The most costly step is calculating the VACF. FFTing that is only the fft
    # of a 1d-array which is fast, even if the length is not a power of two.
    # Padding is not needed.
    #
    natoms = vel.shape[1]
    nstep = vel.shape[0]
    assert vel.shape[-1] == 3, ("last dim of vel must be 3: (nstep,natoms,3)")
    # `c` as "intent(in, out)" could be "intent(out), allocatable" or so,
    # makes extension more pythonic, don't pass `c` in, let be allocated on
    # Fortran side
    c = np.zeros((nstep,), dtype=float)
    if m is None:
        # dummy
        m = np.empty((natoms,), dtype=float)
        use_m = 0
    else:
        use_m = 1
    verbose("calling _flib.vacf ...")
    if nthreads is None:
        # Possible f2py bug workaround: The f2py extension does not always set
        # the number of threads correctly according to OMP_NUM_THREADS. Catch
        # OMP_NUM_THREADS here and set number of threads using the "nthreads"
        # arg.
        key = 'OMP_NUM_THREADS'
        if os.environ.has_key(key):
            nthreads = int(os.environ[key])
            c = _flib.vacf(vel, m, c, method, use_m, nthreads)
        else:            
            c = _flib.vacf(vel, m, c, method, use_m)
    else:        
        c = _flib.vacf(vel, m, c, method, use_m, nthreads)
    verbose("... ready")
    return c


def pdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
         npad=None, tonext=False, mirr=False, method='direct'):
    """Phonon DOS by FFT of the VACF or direct FFT of atomic velocities.
    
    Integral area is normalized to `area`. It is possible (and recommended) to
    zero-padd the velocities (see `npad`). 
    
    Parameters
    ----------
    vel : 3d array (nstep, natoms, 3)
        atomic velocities
    dt : time step
    m : 1d array (natoms,), 
        atomic mass array, if None then mass=1.0 for all atoms is used  
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool 
        use Welch windowing on data before FFT (reduces leaking effect,
        recommended)
    npad : {None, int}
        method='direct' only: Length of zero padding along `axis`. `npad=None`
        = no padding, `npad > 0` = pad by a length of ``(nstep-1)*npad``. `npad
        > 5` usually results in sufficient interpolation.
    tonext : bool
        method='direct' only: Pad `vel` with zeros along `axis` up to the next
        power of two after the array length determined by `npad`. This gives
        you speed, but variable (better) frequency resolution.
    mirr : bool 
        method='vacf' only: mirror one-sided VACF at t=0 before fft

    Returns
    -------
    if full_out = False
        | ``(faxis, pdos)``
        | faxis : 1d array [1/unit(dt)]
        | pdos : 1d array, the phonon DOS, normalized to `area`
    if full_out = True
        | if method == 'direct':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx))``
        | if method == 'vavcf':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))``
        |     fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        |     vacf : 1d array, the VACF
    
    Examples
    --------
    >>> from pwtools.constants import fs,rcm_to_Hz
    >>> tr = Trajectory(...)
    >>> # freq in [Hz] if timestep in [s]
    >>> freq,dos = pdos(tr.velocity, m=tr.mass, dt=tr.timestep*fs, 
    >>>                 method='direct', npad=1)
    >>> # frequency in [1/cm]
    >>> plot(freq/rcm_to_Hz, dos)
    
    Notes
    -----
    padding (only method='direct'): With `npad` we pad the velocities `vel`
    with ``npad*(nstep-1)`` zeros along `axis` (the time axis) before FFT
    b/c the signal is not periodic. For `npad=1`, this gives us the exact
    same spectrum and frequency resolution as with ``pdos(...,
    method='vacf',mirr=True)`` b/c the array to be fft'ed has length
    ``2*nstep-1`` along the time axis in both cases (remember that the
    array length = length of the time axis influences the freq.
    resolution). FFT is only fast for arrays with length = a power of two.
    Therefore, you may get very different fft speeds depending on whether
    ``2*nstep-1`` is a power of two or not (in most cases it won't). Try
    using `tonext` but remember that you get another (better) frequency
    resolution.

    References
    ----------
    [1] Phys Rev B 47(9) 4863, 1993

    See Also
    --------
    :func:`pwtools.signal.fftsample`
    :func:`pwtools.signal.acorr`
    :func:`direct_pdos`
    :func:`vacf_pdos`

    """
    mass = m
    # assume vel.shape = (nstep,natoms,3)
    axis = 0
    assert vel.shape[-1] == 3
    if mass is not None:
        assert len(mass) == vel.shape[1], "len(mass) != vel.shape[1]"
        # define here b/c may be used twice below
        mass_bc = mass[None,:,None]
    if window:
        sl = [None]*vel.ndim 
        sl[axis] = slice(None) # ':'
        vel2 = vel*(welch(vel.shape[axis])[sl])
    else:
        vel2 = vel
    # handle options which are mutually exclusive
    if method == 'vacf':
        assert npad in [0,None], "use npad={0,None} for method='vacf'"
    # padding
    if npad is not None:
        nadd = (vel2.shape[axis]-1)*npad
        if tonext:
            vel2 = pad_zeros(vel2, tonext=True, 
                             tonext_min=vel2.shape[axis] + nadd, 
                             axis=axis)
        else:    
            vel2 = pad_zeros(vel2, tonext=False, nadd=nadd, axis=axis)
    if method == 'direct': 
        full_fft_vel = np.abs(fft(vel2, axis=axis))**2.0
        full_faxis = np.fft.fftfreq(vel2.shape[axis], dt)
        split_idx = len(full_faxis)/2
        faxis = full_faxis[:split_idx]
        # First split the array, then multiply by `mass` and average. If
        # full_out, then we need full_fft_vel below, so copy before slicing.
        arr = full_fft_vel.copy() if full_out else full_fft_vel
        fft_vel = num.slicetake(arr, slice(0, split_idx), axis=axis, copy=False)
        if mass is not None:
            fft_vel *= mass_bc
        # average remaining axes, summing is enough b/c normalization is done below
        # sums: (nstep, natoms, 3) -> (nstep, natoms) -> (nstep,)
        pdos = num.sum(fft_vel, axis=axis, keepdims=True)
        default_out = (faxis, num.norm_int(pdos, faxis, area=area))
        if full_out:
            # have to re-calculate this here b/c we never calculate the full_pdos
            # normally
            if mass is not None:
                full_fft_vel *= mass_bc
            full_pdos = num.sum(full_fft_vel, axis=axis, keepdims=True)
            extra_out = (full_faxis, full_pdos, split_idx)
            return default_out + extra_out
        else:
            return default_out
    elif method == 'vacf':
        vacf = fvacf(vel2, m=mass)
        if mirr:
            fft_vacf = fft(mirror(vacf))
        else:
            fft_vacf = fft(vacf)
        full_faxis = np.fft.fftfreq(fft_vacf.shape[axis], dt)
        full_pdos = np.abs(fft_vacf)
        split_idx = len(full_faxis)/2
        faxis = full_faxis[:split_idx]
        pdos = full_pdos[:split_idx]
        default_out = (faxis, num.norm_int(pdos, faxis, area=area))
        extra_out = (full_faxis, full_pdos, split_idx, vacf, fft_vacf)
        if full_out:
            return default_out + extra_out
        else:
            return default_out


def vacf_pdos(vel, *args, **kwds):
    """Wrapper for ``pdos(..., method='vacf', mirr=True, npad=None)``"""
    if not kwds.has_key('mirr'):
        kwds['mirr'] = True
    return pdos(vel, *args, method='vacf', npad=None, **kwds)


def direct_pdos(vel, *args, **kwds):
    """Wrapper for ``pdos(..., method='direct', npad=1)``"""
    if not kwds.has_key('npad'):
        kwds['npad'] = 1
    if kwds.has_key('pad_tonext'):
        warnings.simplefilter('always')
        warnings.warn("'pad_tonext' was renamed 'tonext'",
            DeprecationWarning)
        kwds['tonext'] = kwds['pad_tonext']
        kwds.pop('pad_tonext')
    return pdos(vel, *args, method='direct', **kwds)
