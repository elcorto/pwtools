#!/usr/bin/env python
# -*- coding: utf-8 -*- 


# Post-process molecular dynamics data produced by the Quantum Espresso package
# (quantum-espresso.org). 
# 
# This module implements the functionallity to calculate the phonon DOS from MD
# trajectories. For parsing output files into a format which is used here, see
# parse.py and test/* for examples.
# 

##from debug import Debug
##DBG = Debug()

# timing of the imports
##DBG.t('import')

import os
from itertools import izip

import numpy as np
norm = np.linalg.norm

# slow import time for these
from scipy.fftpack import fft

# own modules
import constants
import _flib
import common as com
import io
from verbose import verbose
from fft import pad_zeros, welch

# aliases
pjoin = os.path.join

##DBG.pt('import')


# backward compat for older scripts using pwtools, not used here
from crys import coord_trans
from pwscf import atpos_str, atspec_str

#-----------------------------------------------------------------------------
# globals 
#-----------------------------------------------------------------------------

# Used only in verbose().
VERBOSE = False

#-----------------------------------------------------------------------------
# computational
#-----------------------------------------------------------------------------

def velocity(R, dt=None, copy=True, rslice=slice(None)):
    """Compute velocity from 3d array with MD trajectory.
        
    args:
    -----
    R : 3D array, shape (natoms, nstep, 3)
        atomic coords
    dt: float
        time step
    copy : bool
        If False, then we do in-place modification of R to save memory and
        avoid array copies. A view into the modified R is returned.
        Use only if you don't use R after calling this function.
    rslice : slice object, defaults to slice(None), i.e. take all
        a slice for the 2nd axis (time axis) of R  
    
    returns:            
    --------
    V : 3D array, shape (natoms, <determined_by_rslice>, 3)

    notes:
    ------
    Even with copy=False, a temporary copy of R in the calculation made by 
    numpy is unavoidable.
    """
    # FIXME We assume that the time axis the axis=1 in R. This not safe should we
    # ever change that.
    if copy:
        tmp = R.copy()[:,rslice,:]
    else:
        # view into R
        tmp = R[:,rslice,:]
    #FIXME hardcoded time axis
    # Same as tmp[:,1:,:] =  np.diff(tmp, n=1, axis=1).
    tmp[:,1:,:] =  tmp[:,1:,:] - tmp[:,:-1,:]
    # (natoms, nstep, 3), view only, skip j=0 <=> Rold
    V = tmp[:,1:,:]
    verbose("[velocity] V.shape: %s" %repr(V.shape))
    if dt is not None:
        V /= dt
    return V


def pyvacf(V, m=None, method=3):
    """Reference implementation of the VACF of velocities in 3d array `V`. See
    velocity(). We do some numpy vectorization here.
    """
    natoms = V.shape[0]
    nstep = V.shape[1]
    c = np.zeros((nstep,), dtype=float)
    # we add extra multiplications by unity if m is None, but since it's only
    # the ref impl. .. who cares. better than having tons of if's in the loops.
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
                    c[t] += np.dot(V[i,j,:], V[i,j+t,:]) * m[i]
    elif method == 2:    
        # replace 1 inner loop
        for t in xrange(nstep):
            for j in xrange(nstep-t):
                # Multiply with mass-vector m:
                # Use array broadcasting: each col of V[:,j,:] is element-wise
                # multiplied with m (or in other words: multiply each of the 3
                # vectors V[:,j,k] in V[:,j,:] element-wise with m).
                #   V[:,j,:]          -> (natoms, 3)
                #   m[:,np.newaxis]    -> (natoms, 1)
                c[t] += (V[:,j,:] * V[:,j+t,:] * m[:,np.newaxis]).sum()
    elif method == 3:    
        # replace 2 inner loops
        for t in xrange(nstep):
            c[t] = (V[:,:(nstep-t),:] * V[:,t:,:]*m[:,np.newaxis,np.newaxis]).sum()
    else:
        raise ValueError('unknown method: %s' %method)
    # normalize to unity
    c = c / c[0]
    return c


def fvacf(V, m=None, method=2, nthreads=None):
    """Interface to Fortran function _flib.vacf(). Otherwise same
    functionallity as pyvacf(). Use this for production calculations.

    notes:
    ------
    $ python -c "import _flib; print _flib.vacf.__doc__"
    vacf - Function signature:
      c = vacf(v,m,c,method,use_m,[nthreads,natoms,nstep])
    Required arguments:
      v : input rank-3 array('d') with bounds (natoms,nstep,3)
      m : input rank-1 array('d') with bounds (natoms)
      c : input rank-1 array('d') with bounds (nstep)
      method : input int
      use_m : input int
    Optional arguments:
      nthreads : input int
      natoms := shape(v,0) input int
      nstep := shape(v,1) input int
    Return objects:
      c : rank-1 array('d') with bounds (nstep)
    """
    natoms = V.shape[0]
    nstep = V.shape[1]
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
    # With V = np.asarray(V, order='F'), we convert V to F-order and a copy is
    # made. If we don't do it, the f2py wrapper code does. This copy is
    # unavoidable, unless we allocate the array V in F-order in the first
    # place.
    ## c = _flib.vacf(np.asarray(V, order='F'), m, c, method, use_m)
    verbose("calling _flib.vacf ...")
    if nthreads is None:
        # Possible f2py bug workaround: The f2py extension does not always set
        # the number of threads correctly according to OMP_NUM_THREADS. Catch
        # OMP_NUM_THREADS here and set number of threads using the "nthreads"
        # arg.
        key = 'OMP_NUM_THREADS'
        if os.environ.has_key(key):
            nthreads = int(os.environ[key])
            c = _flib.vacf(V, m, c, method, use_m, nthreads)
        else:            
            c = _flib.vacf(V, m, c, method, use_m)
    else:        
        c = _flib.vacf(V, m, c, method, use_m, nthreads)
    verbose("... ready")
    return c


def direct_pdos(V, dt=1.0, m=None, full_out=False, area=1.0, window=True):
    """Compute PDOS without the VACF by direct FFT of the atomic velocities.
    We call this Direct Method. Integral area is normalized 1.0.
    
    args:
    -----
    V : velocity array (natoms, nstep, 3)
    dt : time step in seconds
    m : 1d array (natoms,), atomic mass array, if None then mass=1.0 for all
        atoms is used  
    full_out : bool
    area : float
        normalize area under frequency-PDOS to this value
    window : bool, use Welch windowing on data before FFT (reduces leaking
        effect, recommended)

    returns:
    --------
    full_out = False
        (faxis, pdos)
        faxis : 1d array, frequency in Hz
        pdos : 1d array, the PDOS
    full_out = True
        (faxis, pdos, (full_faxis, full_pdos, split_idx))

    refs:
    -----
    [1] Phys Rev B 47(9) 1993
    """
    massvec = m 
    # FIXME hardcoded time axis
    time_axis = 1
    # fftv: array of VV.shape, axis=1 is the fft of the arrays along axis 1 of
    # VV
    # Pad velocities w/ zeros along `time_axis`.
    # XXX possible optimization: always pad up to the next power of 2
    if window:
        # newaxis stuff
        #   # 1d
        #   >>> a = welch(...)
        #   # tranform to 3d, broadcast to axis 0 and 2
        #   >>> a[None, :, None] 
        sl = [None]*V.ndim
        sl[time_axis] = slice(None)
        VV = V*(welch(V.shape[time_axis])[sl])
    else:
        VV = V
    VV = pad_zeros(VV, nadd=VV.shape[time_axis]-1, axis=time_axis)
    print "fft ..."
    full_fftv = np.abs(fft(VV, axis=time_axis))**2.0
    print "...ready"
    full_faxis = np.fft.fftfreq(VV.shape[time_axis], dt)
    split_idx = len(full_faxis)/2
    faxis = full_faxis[:split_idx]
    # first split the array, then multiply by `massvec` and average
    fftv = slicetake(full_fftv, slice(0, split_idx), axis=time_axis)
    if massvec is not None:
        com.assert_cond(len(massvec) == VV.shape[0], "len(massvec) != VV.shape[0]")
        fftv *= massvec[:,np.newaxis, np.newaxis]
                                 
    # average remaining axes        
    pdos = fftv.sum(axis=0).sum(axis=1)        
    default_out = (faxis, com.norm_int(pdos, faxis, area=area))
    if full_out:
        # have to re-calculate this here b/c we never calculate the full_pdos
        # normally
        if massvec is not None:
            full_pdos = (full_fftv * \
                         massvec[:,np.newaxis, np.newaxis]\
                         ).sum(axis=0).sum(axis=1)
        else:                              
            full_pdos = full_fftv.sum(axis=0).sum(axis=1)
        extra_out = (full_faxis, full_pdos, split_idx)
        return default_out + (extra_out,)
    else:
        return default_out


def vacf_pdos(V, dt=1.0, m=None, mirr=False, full_out=False, area=1.0,
              window=True):
    """Compute PDOS by FFT of the VACF. Integral area is normalized to
    1.0.
    
    args:
    -----
    V : (natoms, nstep, 3)
    dt : time step in seconds
    m : 1d array (natoms,), atomic mass array, if None then mass=1.0 for all
        atoms is used  
    mirr : bool, mirror VACF at t=0 before fft
    full_out : bool
    area : float
        normalize area under frequency-PDOS to this value
    window : bool, use Welch windowing on data before FFT (reduces leaking
        effect, recommended)

    returns:
    --------
    full_out = False
        (faxis, pdos)
        faxis : 1d array, frequency in Hz
        pdos : 1d array, the PDOS
    full_out = True
        (faxis, pdos, (full_faxis, fftcc, split_idx, c))
        ffttc : 1d complex array, result of fft(c) or fft(mirror(c))
        c : 1d array, the VACF
    """
    massvec = m 
    time_axis = 1
    if window:
        sl = [None]*V.ndim
        sl[time_axis] = slice(None)
        VV = V*(welch(V.shape[time_axis])[sl])
    else:
        VV = V
    c = fvacf(VV, m=massvec)
    if mirr:
        verbose("[vacf_pdos] mirror VACF at t=0")
        fftc = fft(mirror(c))
    else:
        fftc = fft(c)
    full_faxis = np.fft.fftfreq(fftc.shape[0], dt)
    full_pdos = np.abs(fftc)
    split_idx = len(full_faxis)/2
    faxis = full_faxis[:split_idx]
    pdos = full_pdos[:split_idx]
    default_out = (faxis, com.norm_int(pdos, faxis, area=area))
    extra_out = (full_faxis, fftc, split_idx, c)
    if full_out:
        return default_out + (extra_out,)
    else:
        return default_out

def mirror(arr):
    """Mirror 1d array `arr`. Length of the returned array is 2*len(arr)-1 ."""
    return np.concatenate((arr[::-1],arr[1:]))


def slicetake(a, sl, axis=None, copy=False):
    """The equivalent of numpy.take(a, ..., axis=<axis>), but accepts slice
    objects instead of an index array. Also by default, it returns a *view* and
    no copy.
    
    args:
    -----
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list or tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : bool, return a copy instead of a view
    
    returns:
    --------
    A view into `a` or copy of a slice of `a`.

    examples:
    ---------
    >>> from numpy import s_
    >>> a = np.random.rand(20,20,20)
    >>> b1 = a[:,:,10:]
    >>> # single slice for axis 2 
    >>> b2 = slicetake(a, s_[10:], axis=2)
    >>> # tuple of slice objects 
    >>> b3 = slicetake(a, s_[:,:,10:])
    >>> (b2 == b1).all()
    True
    >>> (b3 == b1).all()
    True
    """
    # The long story
    # --------------
    # 
    # 1) Why do we need that:
    # 
    # # no problem
    # a[5:10:2]
    # 
    # # the same, more general
    # sl = slice(5,10,2)
    # a[sl]
    #
    # But we want to:
    #  - Define (type in) a slice object only once.
    #  - Take the slice of different arrays along different axes.
    # Since numpy.take() and a.take() don't handle slice objects, one would
    # have to use direct slicing and pay attention to the shape of the array:
    #       
    #     a[sl], b[:,:,sl,:], etc ...
    # 
    # We want to use an 'axis' keyword instead. np.r_() generates index arrays
    # from slice objects (e.g r_[1:5] == r_[s_[1:5] ==r_[slice(1,5,None)]).
    # Since we need index arrays for numpy.take(), maybe we can use that? Like
    # so:
    #     
    #     a.take(r_[sl], axis=0)
    #     b.take(r_[sl], axis=2)
    # 
    # Here we have what we want: slice object + axis kwarg.
    # But r_[slice(...)] does not work for all slice types. E.g. not for
    #     
    #     r_[s_[::5]] == r_[slice(None, None, 5)] == array([], dtype=int32)
    #     r_[::5]                                 == array([], dtype=int32)
    #     r_[s_[1:]]  == r_[slice(1, None, None)] == array([0])
    #     r_[1:]
    #         ValueError: dimensions too large.
    # 
    # The returned index arrays are wrong (or we even get an exception).
    # The reason is given below. 
    # Bottom line: We need this function.
    #
    # The reason for r_[slice(...)] gererating sometimes wrong index arrays is
    # that s_ translates a fancy index (1:, ::5, 1:10:2, ...) to a slice
    # object. This *always* works. But since take() accepts only index arrays,
    # we use r_[s_[<fancy_index>]], where r_ translates the slice object
    # prodced by s_ to an index array. THAT works only if start and stop of the
    # slice are known. r_ has no way of knowing the dimensions of the array to
    # be sliced and so it can't transform a slice object into a correct index
    # array in case of slice(<number>, None, None) or slice(None, None,
    # <number>).
    #
    # 2) Slice vs. copy
    # 
    # numpy.take(a, array([0,1,2,3])) or a[array([0,1,2,3])] return a copy of
    # `a` b/c that's "fancy indexing". But a[slice(0,4,None)], which is the
    # same as indexing (slicing) a[:4], return *views*. 
    
    if axis is None:
        slices = sl
    else: 
        # Note that these are equivalent:
        #   a[:]
        #   a[s_[:]] 
        #   a[slice(None)] 
        #   a[slice(None, None, None)]
        #   a[slice(0, None, None)]   
        slices = [slice(None)]*a.ndim
        slices[axis] = sl
    # a[...] can take a tuple or list of slice objects
    # a[x:y:z, i:j:k] is the same as
    # a[(slice(x,y,z), slice(i,j,k))] == a[[slice(x,y,z), slice(i,j,k)]]
    if copy:
        return a[slices].copy()
    else:        
        return a[slices]


def sliceput(a, b, sl, axis=None):
    """The equivalent of a[<slice or index>]=b, but accepts slices objects
    instead of array indices or fancy indexing (e.g. a[:,1:]).
    
    args:
    -----
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list or tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    
    returns:
    --------
    The modified `a`.
    
    examples:
    ---------
    >>> from numpy import s_
    >>> a=np.arange(12).reshape((2,6))
    >>> a[:,1:3] = 100
    >>> a
    array([[  0, 100, 100,   3,   4,   5],
           [  6, 100, 100,   9,  10,  11]])
    >>> sliceput(a, 200, s_[1:3], axis=1)
    array([[  0, 200, 200,   3,   4,   5],
           [  6, 200, 200,   9,  10,  11]])
    >>> sliceput(a, 300, s_[:,1:3])
    array([[  0, 300, 300,   3,   4,   5],
           [  6, 300, 300,   9,  10,  11]])
    """
    if axis is None:
        # silce(...) or (slice(...), slice(...), ...)
        tmp = sl
    else:
        # [slice(...), slice(...), ...]
        tmp = [slice(None)]*len(a.shape)
        tmp[axis] = sl
    a[tmp] = b
    return a

