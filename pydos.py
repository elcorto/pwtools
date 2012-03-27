# pydos.py
#
# This module implements the functionallity to calculate the phonon density of
# states (PDOS) from MD trajectories. For parsing output files into a format
# which is used here, see parse.py and test/* for examples. For a theory
# overview, see README and refs therein.

import os
import numpy as np
from scipy.fftpack import fft
import constants, _flib, num
from pwtools.verbose import verbose
from pwtools.signal import pad_zeros, welch

def velocity(coords, dt=None, axis=0):
    """Compute velocity from 3d array with MD trajectory by simple finite
    differences.
        
    args:
    -----
    coords : 3d array
        Cartesian atomic coords of an MD trajectory. The time axis is defined
        by "axis". Along this axis, 2d arrays (natoms,3) are expected.
    dt: optional, float
        time step
    axis : optional, int
        Time axis of "coords".

    returns:            
    --------
    vel : 3D array
        Usally, this is (nstep-1,natoms, 3)
    """
    vel = np.diff(coords, n=1, axis=axis)
    if dt is not None:
        vel /= dt
    return vel


def pyvacf(vel, m=None, method=3):
    """Reference implementation for calculating the VACF of velocities in 3d
    array `vel`. This is slow. Use for debugging only. For production, use
    fvacf().
    
    args:
    -----
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        Which method to use.
    
    returns:
    --------
    c : 1d array (nstep,)
        VACF
    """
    natoms = vel.shape[1]
    nstep = vel.shape[0]
    c = np.zeros((nstep,), dtype=float)
    # We add extra multiplications by unity if m is None, but since it's only
    # the ref impl. .. who cares. Better than having tons of if's in the loops.
    if m is None:
        m = np.ones((natoms,), dtype=float)

    if method == 1:
        # c(t) = <v(t0) v(t0 + t)> / <v(t0)**2> = C(t) / C(0)
        #
        # "displacements" `t'
        for t in xrange(nstep):
            # time origins t0 == j
            for j in xrange(nstep-t):
                for i in xrange(natoms):
                    c[t] += np.dot(vel[j,i,:], vel[j+t,i,:]) * m[i]
    elif method == 2:    
        # replace 1 inner loop
        for t in xrange(nstep):
            for j in xrange(nstep-t):
                # Multiply with mass-vector m, use broadcasting.
                # (natoms, 3) * (natoms, 1) -> (natoms, 3)
                c[t] += (vel[j,...] * vel[j+t,...] * m[:,None]).sum()
    elif method == 3:    
        # replace 2 inner loops:
        # (xx, natoms, 3) * (1, natoms, 1) -> (xx, natoms, 3)
        for t in xrange(nstep):
            c[t] = (vel[:(nstep-t),...] * vel[t:,...]*m[None,:,None]).sum()
    else:
        raise ValueError('unknown method: %s' %method)
    # normalize to unity
    c = c / c[0]
    return c


def fvacf(vel, m=None, method=2, nthreads=None):
    """Interface to Fortran function _flib.vacf(). Otherwise same
    functionallity as pyvacf(). Use this for production calculations.
    
    args:
    -----
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        Which method to use.
    nthreads : int ot None
        If int, then use this many OpenMP threads in the Fortran extension.
        Only useful if the extension was compiled with OpenMP support, of
        course.

    returns:
    --------
    c : 1d array (nstep,)
        VACF

    notes:
    ------
    Fortran extension:
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

    see also:          
    ---------
    _flib
    vacf_pdos()
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
    assert vel.shape[-1] == 3
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
    # (nstep, natoms, 3) -> (natoms, 3, nstep)
    vel_f = np.rollaxis(vel, 0, 3)
    assert vel_f.shape == (natoms, 3, nstep)
    verbose("calling _flib.vacf ...")
    if nthreads is None:
        # Possible f2py bug workaround: The f2py extension does not always set
        # the number of threads correctly according to OMP_NUM_THREADS. Catch
        # OMP_NUM_THREADS here and set number of threads using the "nthreads"
        # arg.
        key = 'OMP_NUM_THREADS'
        if os.environ.has_key(key):
            nthreads = int(os.environ[key])
            c = _flib.vacf(vel_f, m, c, method, use_m, nthreads)
        else:            
            c = _flib.vacf(vel_f, m, c, method, use_m)
    else:        
        c = _flib.vacf(vel_f, m, c, method, use_m, nthreads)
    verbose("... ready")
    return c


def direct_pdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
                pad_tonext=False, axis=0):
    """Phonon DOS without the VACF by direct FFT of the atomic velocities.
    We call this Direct Method. Integral area is normalized to "area".
    
    args:
    -----
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
    pad_tonext : bool
        Pad `vel` with zeros along `axis` up to the next power of two after
        2*nstep-1. This gives you speed, but variable (better) frequency
        resolution.
    axis : int
        Time axis of "vel".

    returns:
    --------
    full_out = False
        (faxis, pdos)
        faxis : 1d array [1/unit(dt)]
        pdos : 1d array, the PDOS
    full_out = True
        (faxis, pdos, (full_faxis, full_pdos, split_idx))
    
    notes:
    ------
    padding: By default we pad the velocities `vel` with nstep-1 zeros along
        axis (the time axis) before FFT b/c the signal is not periodic. This
        gives us the exact same frequency resolution as with vacf_pdos(...,
        mirr=True) b/c the array to be fft'ed has length 2*nstep-1 along the
        time axis in both cases (remember that the array length = length of the
        time axis influences the freq. resolution). FFT is only fast for arrays
        with length = a power of two. Therefore, you may get very different fft
        speeds depending on whether 2*nstep-1 is a power of two or not (in most
        cases it won't). Try using `pad_tonext` but remember that you get
        another (better) frequency resolution.

    axis : That this is not completely transparent as we don't use smth like
        atomaxis=1. But if we change the assumed shape of `vel`, then we don't
        have to change much code, b/c most operations take place along the time
        axis (`axis`) already and are coded that way.

    refs:
    -----
    [1] Phys Rev B 47(9) 4863, 1993

    see also:
    ---------
    vacf_pdos
    pwtools.signal.fftsample
    """
    mass = m
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
    if pad_tonext:
        vel2 = pad_zeros(vel2, tonext=True, tonext_min=vel2.shape[axis]*2 - 1, axis=axis)
    else:
        vel2 = pad_zeros(vel2, nadd=vel2.shape[axis]-1, axis=axis)
    verbose("fft ...")
    full_fft_vel = np.abs(fft(vel2, axis=axis))**2.0
    verbose("...ready")
    full_faxis = np.fft.fftfreq(vel2.shape[axis], dt)
    split_idx = len(full_faxis)/2
    faxis = full_faxis[:split_idx]
    # First split the array, then multiply by `mass` and average. If
    # full_out, then we need full_fft_vel below, so copy before slicing.
    arr = full_fft_vel.copy() if full_out else full_fft_vel
    fft_vel = num.slicetake(arr, slice(0, split_idx), axis=axis)
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


def vacf_pdos(vel, dt=1.0, m=None, mirr=False, full_out=False, area=1.0,
              window=True, axis=0):
    """Phonon DOS by FFT of the VACF. Integral area is normalized to
    "area".
    
    args:
    -----
    vel : 3d array (nstep, natoms, 3)
        atomic velocities
    dt : time step
    m : 1d array (natoms,), 
        atomic mass array, if None then mass=1.0 for all atoms is used  
    mirr : bool 
        mirror VACF at t=0 before fft
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool 
        Use Welch windowing on data before FFT (reduces leaking effect,
        recommended).
    axis : int
        Time axis of "vel". 
    
    notes:
    ------
    axis : That this is not completely transparent as we don't use smth like
        atomaxis=1. But if we change the assumed shape of `vel`, then we don't
        have to change much code, b/c most operations take place along the time
        axis (`axis`) already and are coded that way.

    returns:
    --------
    full_out = False
        (faxis, pdos)
        faxis : 1d array [1/unit(dt)]
        pdos : 1d array, the PDOS
    full_out = True
        (faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))
        fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        vacf : 1d array, the VACF
    """
    mass = m 
    assert vel.shape[-1] == 3
    if window:
        sl = [None]*vel.ndim
        sl[axis] = slice(None)
        vel2 = vel*(welch(vel.shape[axis])[sl])
    else:
        vel2 = vel
    vacf = fvacf(vel2, m=mass)
    if mirr:
        verbose("[vacf_pdos] mirror VACF at t=0")
        fft_vacf = fft(mirror(vacf))
    else:
        fft_vacf = fft(vacf)
    full_faxis = np.fft.fftfreq(fft_vacf.shape[0], dt)
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


def mirror(arr):
    """Mirror 1d array `arr`. Length of the returned array is 2*len(arr)-1 ."""
    return np.concatenate((arr[::-1],arr[1:]))

