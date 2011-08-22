# pydos.py
#
# This module implements the functionallity to calculate the phonon density of
# states (PDOS) from MD trajectories. For parsing output files into a format
# which is used here, see parse.py and test/* for examples. For a theory
# overview, see README and refs therein.

import os
from itertools import izip

import numpy as np
norm = np.linalg.norm

# slow import time for these
from scipy.fftpack import fft

from pwtools import constants
from pwtools import _flib
from pwtools import common as com
from pwtools import io
from pwtools.verbose import verbose
from pwtools.signal import pad_zeros, welch
from pwtools import num

# aliases
pjoin = os.path.join


# backward compat for older scripts using pwtools, not used here
from pwtools.crys import coord_trans
from pwtools.pwscf import atpos_str, atspec_str
from pwtools.num import slicetake, sliceput

#-----------------------------------------------------------------------------
# computational
#-----------------------------------------------------------------------------

def velocity(coords, dt=None, copy=True, tslice=slice(None), axis=-1):
    """Compute velocity from 3d array with MD trajectory by simple finite
    differences.
        
    args:
    -----
    coords : 3d array (natoms, 3, nstep)
        Cartesian atomic coords of an MD trajectory. The time axis is defined
        by "axis". Along this axis, 2d arrays (natoms,3) are expected.
    dt: optional, float
        time step
    copy : optional, bool
        If False, then we do in-place modification of coords to save memory and
        avoid array copies. A view into the modified coords is returned.
        Use only if you don't use coords after calling this function.
    tslice : optional, slice object 
        Defaults to slice(None), i.e. take all entries along the time axis
        "axis" of "coords".  
    axis : optional, int
        Time axis of "coords".

    returns:            
    --------
    vel : 3D array, shape (natoms, 3, <determined_by_tslice>) 
        Usally, this is (natoms, 3, nstep-1) for tslice=slice(None), i.e. if
        vel is computed from all steps.

    notes:
    ------
    Even with copy=False, a temporary copy of coords in the calculation made by 
    numpy is unavoidable.
    """
    # View or copy of coords, optionally sliced by "tslice" along "axis".
    tmpsl = [slice(None)]*3
    tmpsl[axis] = tslice
    if copy:
        tmp = coords.copy()[tmpsl]
    else:
        # view
        ##tmp = coords[tmpsl]
        tmp = coords
    # view into tmp       
    vel = slicetake(tmp, sl=np.s_[1:], axis=axis, copy=False)
    # vel[:] to put the result into the memory of "vel", otherwise, vel is a
    # new assignment ans thus a new array
    vel[:] = np.diff(tmp, n=1, axis=axis)
    verbose("[velocity] velocity shape: %s" %repr(vel.shape))
    if dt is not None:
        vel /= dt
    return vel


def pyvacf(vel, m=None, method=3):
    """Reference implementation for calculating the VACF of velocities in 3d
    array `vel`.
    
    args:
    -----
    vel : 3d array, (natoms, 3, nstep)
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
    natoms = vel.shape[0]
    nstep = vel.shape[-1]
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
                    c[t] += np.dot(vel[i,:,j], vel[i,:,j+t]) * m[i]
    elif method == 2:    
        # replace 1 inner loop
        for t in xrange(nstep):
            for j in xrange(nstep-t):
                # Multiply with mass-vector m:
                # Use array broadcasting: each col of vel[:,:,j] is element-wise
                # multiplied with m (or in other words: multiply each of the
                # k=1,2,3 vectors vel[:,k,j] in vel[:,:,j] element-wise with m).
                c[t] += (vel[...,j] * vel[...,j+t] * m[:,np.newaxis]).sum()
    elif method == 3:    
        # replace 2 inner loops
        for t in xrange(nstep):
            c[t] = (vel[...,:(nstep-t)] * vel[...,t:]*m[:,np.newaxis,np.newaxis]).sum()
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
    vel : 3d array, (natoms, 3, nstep)
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

    see also:          
    ---------
    _flib
    direct_pdos
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
    natoms = vel.shape[0]
    nstep = vel.shape[-1]
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


def direct_pdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
                pad_tonext=False, axis=-1):
    """Compute PDOS without the VACF by direct FFT of the atomic velocities.
    We call this Direct Method. Integral area is normalized to "area".
    
    args:
    -----
    vel : 3d array (natoms, 3, nstep)
        atomic velocities
    dt : time step in seconds
    m : 1d array (natoms,), 
        atomic mass array, if None then mass=1.0 for all atoms is used  
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool 
        use Welch windowing on data before FFT (reduces leaking effect,
        recommended)
    pad_tonext : bool
        Padd `vel` with zeros along `axis` up to the next power of two after
        2*nstep-1. This gives you speed, but variable (better) frequency
        resolution.
    axis : int
        Time axis of "vel".

    returns:
    --------
    full_out = False
        (faxis, pdos)
        faxis : 1d array, frequency in Hz
        pdos : 1d array, the PDOS
    full_out = True
        (faxis, pdos, (full_faxis, full_pdos, split_idx))
    
    notes:
    ------
    By default we pad the velocities `vel` with nstep-1 zeros along axis=2
    (the time axis) before FFT b/c the signal is not periodic. This gives us
    the exact same frequency resolution as with vacf_pdos(..., mirr=True) b/c
    the array to be fft'ed has length 2*nstep-1 along the time axis in both
    cases (remember that the array length = length of the time axis influences
    the freq. resolution). FFT is only fast for arrays with length = a power of
    two. Therefore, you may get very different fft speeds depending on whether
    2*nstep-1 is a power of two or not (in most cases it won't). Try using
    `pad_tonext` but remember that you get another (better) frequency
    resolution.

    refs:
    -----
    [1] Phys Rev B 47(9) 4863, 1993

    see also:
    ---------
    vacf_pdos
    pwtools.signal.fftsample
    """
    # * fft_vel: array of vel2.shape, axis="axis" is the fft of the arrays along
    #   axis 1 of vel2
    # * using broadcasting for multiplication with Welch window:
    #   # 1d
    #   >>> a = welch(...)
    #   # tranform to 3d, broadcast to axis 0 and 1 (time axis = 2)
    #   >>> a[None, None, :] # None == np.newaxis
    massvec = m 
    if window:
        sl = [None]*vel.ndim 
        sl[axis] = slice(None)
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
    # First split the array, then multiply by `massvec` and average. If
    # full_out, then we need full_fft_vel below, so copy before slicing.
    arr = full_fft_vel.copy() if full_out else full_fft_vel
    fft_vel = slicetake(arr, slice(0, split_idx), axis=axis)
    if massvec is not None:
        com.assert_cond(len(massvec) == vel2.shape[0], "len(massvec) != vel2.shape[0]")
        fft_vel *= massvec[:,np.newaxis, np.newaxis]
    # average remaining axes (axis 0 and 1), summing is enough b/c
    # normalization is done below      
    pdos = fft_vel.sum(axis=0).sum(axis=0)        
    default_out = (faxis, num.norm_int(pdos, faxis, area=area))
    if full_out:
        # have to re-calculate this here b/c we never calculate the full_pdos
        # normally
        if massvec is not None:
            full_fft_vel *= massvec[:,np.newaxis, np.newaxis]
        full_pdos = full_fft_vel.sum(axis=0).sum(axis=0)
        extra_out = (full_faxis, full_pdos, split_idx)
        return default_out + extra_out
    else:
        return default_out


def vacf_pdos(vel, dt=1.0, m=None, mirr=False, full_out=False, area=1.0,
              window=True, axis=-1):
    """Compute PDOS by FFT of the VACF. Integral area is normalized to
    "area".
    
    args:
    -----
    vel : 3d array (natoms, 3, nstep)
        atomic velocities
    dt : time step in seconds
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

    returns:
    --------
    full_out = False
        (faxis, pdos)
        faxis : 1d array, frequency in Hz
        pdos : 1d array, the PDOS
    full_out = True
        (faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))
        fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        vacf : 1d array, the VACF
    """
    massvec = m 
    if window:
        sl = [None]*vel.ndim
        sl[axis] = slice(None)
        vel2 = vel*(welch(vel.shape[axis])[sl])
    else:
        vel2 = vel
    vacf = fvacf(vel2, m=massvec)
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



