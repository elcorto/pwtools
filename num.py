# num.py : numpy/scipy like stuff.

import os, copy
import types
import itertools
import numpy as np
from math import sqrt
import scipy.optimize as optimize
from scipy.interpolate import bisplrep, \
    bisplev, splev, splrep
from pwtools import _flib   

# Hack for older scipy versions.
try: 
    from scipy.interpolate import CloughTocher2DInterpolator
except ImportError:
    # Don't throw a warning here b/c (1) this module is imported often and that
    # would annoy anyone to no end and (2) this interpolator isn't used much,
    # only for experimentation. It's enough to fail inside Interpol2D() if
    # needed.
    CloughTocher2DInterpolator = None


def normalize(a):
    """Normalize array by it's max value. Works also for complex arrays.

    Examples
    --------
    >>> a=np.array([3+4j, 5+4j])
    >>> a
    array([ 3.+4.j,  5.+4.j])
    >>> a.max()
    (5.0+4.0j)
    >>> a/a.max()
    array([ 0.75609756+0.19512195j,  1.00000000+0.j ])
    """
    return a / a.max()

def vlinspace(a, b, num, endpoint=True):
    """Like numpy.linspace, but for 1d arrays. Generate uniformly spaced points
    (vectors) along the distance vector connecting a and b.
    
    Parameters
    ----------
    a, b : 1d arrays
    num : int

    Returns
    -------
    array (num, len(a)), each row is a "point" between `a` and `b`
    """
    assert a.ndim == b.ndim == 1, "expect 1d arrays"
    assert len(a) == len(b), "`a` and `b` must have equal length"
    assert num >= 1, "`num` must be >= 1"
    # distance vec connecting a and b
    dv = b-a
    if endpoint:
        # If num == 1, then the value of `ddv` doesn't matter b/c ret == a.
        ddv = 0 if (num == 1) else dv/float(num-1)
    else:        
        ddv = dv/float(num)
    ret = np.empty((num, len(dv)), dtype=float)
    ret[...] = ddv
    ret[0,:] = a
    return np.cumsum(ret, axis=0)
    
def norm_int(y, x, area=1.0):
    """Normalize integral area of y(x) to `area`.
    
    Parameters
    ----------
    x,y : numpy 1d arrays
    area : float

    Returns
    -------
    scaled y

    Notes
    -----
    The argument order y,x might be confusing. x,y would be more natural but we
    stick to the order used in the scipy.integrate routines.
    """
    from scipy.integrate import simps
    # First, scale x and y to the same order of magnitude before integration.
    # This may be necessary to avoid numerical trouble if x and y have very
    # different scales.
    fx = 1.0 / np.abs(x).max()
    fy = 1.0 / np.abs(y).max()
    sx = fx*x
    sy = fy*y
##    # Don't scale.
##    fx = fy = 1.0
##    sx, sy = x, y
    # Area under unscaled y(x).
    _area = simps(sy, sx) / (fx*fy)
    return y*area/_area

# XXX don't use, implement central diffs
def deriv_fd(y, x=None, n=1):
    """n-th derivative for 1d arrays of possibly nonuniformly sampled data.
    Returns matching x-axis for plotting. Simple finite differences are used:
    f'(x) = [f(x+h) - f(x)] / h
    
    Parameters
    ----------
    x,y : 1d arrays of same length
        if x=None, then x=arange(len(y)) is used
    n : int
        order of the derivative
    
    Returns
    -------
    xd, yd
    xd : 1d array, (len(x)-n,)
        matching x-axis
    yd : 1d array, (len(x)-n,)
        n-th derivative of y at points xd

    Notes
    -----
    n > 1 (e.g. n=2 -> 2nd derivative) is done by
    recursive application. 
    
    For nonuniformly sampled data, errors blow up quickly. You are strongly
    engouraged to re-sample the data with constant h (e.g. by spline
    interpolation first). Then, derivatives up to ~ 4th order are OK for
    plotting, not for further calculations (unless h is *very* small)!.
    If you need very accurate derivatives, look into NR, 3rd ed., ch. 5.7 and
    maybe scipy.derivative(). 

    Each application returns len(x)-1 points. So for n=3, the returned x and y
    have len(x)-3.

    Examples
    --------
    >>> x=sort(rand(100)*10); y=sin(x); plot(x,y, 'o-'); plot(x,cos(x), 'o-')
    >>> x1,y1=deriv_fd(y,x,1) # cos(x)
    >>> x2,y2=deriv_fd(y,x,2) # -sin(x)
    >>> plot(x1, y1, lw=2) # cos(x)
    >>> plot(x2, -y2, lw=2) # sin(x)
    >>> x=linspace(0,10,100); y=sin(x); plot(x,y, 'o-'); plot(x,cos(x), 'o-')
    >>> ...
    
    See Also
    --------
    numpy.diff()
    numpy.gradient()
    """
    assert n > 0, "n <= 0 makes no sense"
    if n > 1:
        x,y = deriv_fd(y, x, n=1)
        return deriv_fd(y, x, n=n-1)
    else:            
        if x is None:
            x = np.arange(len(y))
        dx = np.diff(x)
        return x[:-1]+.5*dx, np.diff(y)/dx


def deriv_spl(y, x=None, xnew=None, n=1, fullout=True, **splrep_kwargs):
    """n-th derivative for 1d arrays of possibly nonuniformly sampled data.
    Returns matching x-axis for plotting. Splines are used.
    
    Parameters
    ----------
    x,y : 1d arrays of same length
        if x=None, then x=arange(len(y)) is used
    xnew : {None, 1d array)
        x-axis to evaluate the derivative, if None then xnew=x
    n : int
        order of the derivative, can only be <= k 
    fullout : bool
        return xd, yd or just yd
    splrep_kwargs : keyword args to scipy.interpolate.splrep, default: k=3, s=0

    Returns
    -------
    if fullout:
        xd, yd
    else:
        yd
    xd : 1d array, (len(x) or len(xnew),)
    yd : 1d array, (len(x) or len(xnew),)
        n-th derivative of y at points xd
    
    Notes
    -----
    xd is actually == x or xnew (if x is not None). xd can be returned to match
    the function signature of deriv_fd.
    """
    assert n > 0, "n <= 0 makes no sense"
    if x is None:
        x = np.arange(len(y))
    if xnew is None:
        xnew = x
    for key, val in {'s':0, 'k':3}.iteritems():
        if not splrep_kwargs.has_key(key):
            splrep_kwargs[key] = val
    yd = splev(xnew, splrep(x, y, **splrep_kwargs), der=n)
    if fullout:
        return xnew, yd
    else:
        return yd

def _splroot(x, y, der=0):
    # helper for find{min,root}
    tck = splrep(x, y, k=3, s=0)
    func = lambda xx: splev(xx, tck, der=der)
    x0 = optimize.brentq(func, x[0], x[-1])
    return np.array([x0, splev(x0, tck)])


def findmin(x, y):
    """Find minimum of x-y curve by searching for the root of the 1st
    derivative of a spline thru x,y. `x` must be sorted min -> max and the
    interval [x[0], x[-1]] must contain the minimum.
    
    This is intended for quick interactive work. For working with
    pre-calculated splines, see Spline.

    Parameters
    ----------
    x,y : 1d arrays

    Returns
    -------
    array([x0, y(x0)])
    """
    return _splroot(x, y, der=1)


def findroot(x, y):
    """Find root of x-y curve by searching for the root of a spline thru x,y.
    `x` must be sorted min -> max and the interval [x[0], x[-1]] must contain
    the root.

    This is intended for quick interactive work. For working with
    pre-calculated splines, see Spline.
    
    Parameters
    ----------
    x,y : 1d arrays

    Returns
    -------
    array([x0, y(x0)])
    """
    return _splroot(x, y, der=0)


class Spline(object):
    """Like scipy.interpolate.UnivariateSpline, this is a wrapper around
    scipy.interpolate.splrep/splev with some nice features like y->x lookup and
    interpolation accuracy check etc. It basically simplifies setting up a
    spline interpolation and holds x-y data plus the spline knots (self.tck)
    together in one place. You can work with the methods here, but you can also
    use the normal tck (self.tck) in scipy.interpolate.splev() etc.

    Examples
    --------
    >>> from scipy.interpolate import splev
    >>> x = linspace(0,10,100)
    >>> y = sin(x)
    >>> sp = Spline(x,y)
    >>> plot(x,y)
    >>> plot(x, sp(x))
    >>> plot(x, sp.splev(x))      # the same
    >>> plot(x, splev(x, sp.tck)) # the same
    >>> plot(x, sp(x, der=1), label='1st derivative')
    >>> xx = sp.invsplev(0.5, xab=[0, pi/2])
    >>> print("at %f degrees, sin(x) = 0.5" %(xx/pi*180))
    >>> 
    >>> y = x**2 - 5
    >>> sp = Spline(x,y)
    >>> print("the root is at x=%f" %sp.invsplev(0.0))
    """
    def __init__(self, x, y, eps=1e-10, checkeps=True, **splrep_kwargs):
        """
        Parameters
        ----------
        x, y : numpy 1d arrays
        eps : float
            Accuracy threshold. Spline must interpolate points with an error
            less then eps. Useless if you use splrep(...,s=..) with "s" (the
            smoothing factor) much bigger than 0. See `ckeckeps`.
        checkeps : bool
            Whether to use `eps` to ckeck interpolation accuracy.
        **splrep_kwargs : keywords args to splrep(), default: k=3, s=0            
        """
        self.x = x
        self.y = y
        self.eps = eps
        assert (np.diff(self.x) >= 0.0).all(), ("x wronly ordered")
        self.splrep_kwargs = {'s':0, 'k':3}
        self.splrep_kwargs.update(splrep_kwargs)
        self.tck = splrep(self.x, self.y, **self.splrep_kwargs)
        if checkeps:
            err = np.abs(self.splev(self.x) - self.y)
            assert (err < self.eps).all(), \
                    ("spline not accurate to eps=%e, max(error)=%e, raise eps"\
                    %(self.eps, err.max()))

    def __call__(self, *args, **kwargs):
        return self.splev(*args, **kwargs)

    def _findroot(self, func, x0=None, xab=None):
        """Find root of `func` by Newton's method if `x0` is given or Brent's
        method if `xab` is given. If neither is given, then
        ``xab=[self.x[0],self.x[-1]]`` and Brent's method is used.

        Parameters
        ----------
        func : callable, must accept a scalar and retun a scalar
        x0 : float
            start guess for Newton's secant method
        xab : sequence of length 2
            start bracket for Brent's method, root must lie in between
        
        Returns
        -------
        xx : scalar
            the root of func(x)
        """
        if x0 is not None:
            xx = optimize.newton(func, x0)
        else:
            if xab is None:
                xab = [self.x[0], self.x[-1]]
            xx = optimize.brentq(func, xab[0], xab[1])
        return xx    
    
    def is_mono(self):
        """Return True if the curve described by the spline is monotonic."""
        tmp = np.diff(np.sign(np.diff(self.splev(self.x))))       
        return (tmp == 0).all()

    def splev(self, x, *args, **kwargs):
        return splev(x, self.tck, *args, **kwargs)

    def invsplev(self, y0, x0=None, xab=None):
        """Lookup x for a given y, i.e. "inverse spline evaluation", hence
        the name. Find x where y(x) == y0 by calculating the root of y(x) -
        y0. We can use Newton's (x0) or Brent's (xab) methods. Use only one of
        them. If neither is given, we use xab=[x[0], x[-1]] and Brent.
       
        There are a few caveats. The result depends the nature of the
        underlying x-y curve (is it strictly monotinic -> hopefully one root or
        oscillating -> possibly several roots) and on how good x0 or xab are.
        For instance, a bad x0 will cause the Newton method to converge to a
        different root (if there is one) or to converge not at all. Or an
        interval xab which contains no (or several) root(s) will cause the
        Brent method to error out or give wrong/unexpected results. Always plot
        you data before using.
        
        Works only for scalar input (one point lookup). For many points, try to
        construct an inverse spline: Spline(y,x).
        
        Parameters
        ----------
        x0 : float
            start guess for Newton's secant method
        xab : sequence of length 2
            start bracket for Brent's method, root must lie in between
        
        Returns
        -------
        xx : scalar
        """
        # The other possibility to implement this is to construct an inverse
        # spline Spline(y,x) and do the lookup via splev(y0, ...). But this
        # requires the data x,y to be monotonic b/c otherwise, the lookup y->x
        # is not unique. Here, the user is responsible for providing a
        # meaningful x0 / xab, which is more flexible and generic.
        ymn, ymx = self.y.min(), self.y.max()
        assert (ymn <= y0 <= ymx), ("y0 (%e) outside y data range [%e, %e]"
                                    %(y0, ymn, ymx))
        func = lambda x: self.splev(x) - y0
        return self._findroot(func, x0=x0, xab=xab)
   
    def get_min(self, x0=None, xab=None):
        """Return x where y(x) = min(y) by calculating the root of the
        spline's 1st derivative.
        
        Parameters
        ----------
        x0 or xab: see self.invsplev()
        
        Returns
        -------
        xx : scalar
            min(y) = y(xx)
        """
        func = lambda x: self.splev(x, der=1)
        return self._findroot(func, x0=x0, xab=xab)
    
    def get_root(self, x0=None, xab=None):
        """Return x where y(x) = 0 by calculating the root of the spline.
        This function is actually redundant b/c it can be done with
        self.invsplev(0.0, ...), i.e. lookup x where y=0, which is exactly the
        root. But we keep it for reference and convenience.
        
        Parameters
        ----------
        x0 or xab: see self.invsplev()
        
        Returns
        -------
        xx : scalar
            y(xx) = 0
        """
        return self._findroot(self.splev, x0=x0, xab=xab)
    

def slicetake(a, sl, axis=None, copy=False):
    """The equivalent of numpy.take(a, ..., axis=<axis>), but accepts slice
    objects instead of an index array. Also by default, it returns a *view* and
    no copy.
    
    Parameters
    ----------
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list or tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : bool, return a copy instead of a view
    
    Returns
    -------
    A view into `a` or copy of a slice of `a`.

    Examples
    --------
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
    >>> # simple extraction too, sl = integer
    >>> (a[...,5] == slicetake(a, 5, axis=-1))
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
    
    Parameters
    ----------
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list or tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    
    Returns
    -------
    The modified `a`.
    
    Examples
    --------
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


def extend_array(arr, nstep, axis=0):
    """Repeat an array along ``axis`` by inserting a new axis (dimension)
    before ``axis``. Use this to "broadcast" e.g. a 2d array (3,3) ->
    (3,3,nstep).
    
    Parameters
    ----------
    arr : ndarray
    nstep : int, number of times to repeat
    axis : axis to add
    
    Examples
    --------
    >>> a=arange(4)
    >>> extend_array(a, 3, 0)
    array([[0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3]])
    >>> extend_array(a, 3, 0).shape
    (3, 4)
    >>> extend_array(a, 3, 1)
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> extend_array(a, 3, 1).shape
    (4, 3)
    >>> a=arange(4).reshape(2,2)
    >>> extend_array(a, 3, 0).shape
    (3, 2, 2)
    >>> extend_array(a, 3, 1).shape
    (2, 3, 2)
    >>> extend_array(a, 3, 2).shape
    (2, 2, 3)
    >>> extend_array(a, 3, 2)[...,0]
    array([[0, 1],
           [2, 3]])
    >>> extend_array(a, 3, 2)[...,1]
    array([[0, 1],
           [2, 3]])
    >>> extend_array(a, 3, 2)[...,2]
    array([[0, 1],
           [2, 3]])
    
    See Also
    --------
    np.repeat()
    """
    # XXX Make more effective by using stride_tricks, see
    # http://thread.gmane.org/gmane.comp.python.numeric.general/48096 .
    # Test if this survives pickle / unpickle. Probably not.
    #
    # Also, maybe add attr 'extended' to tha array. setattr() doesn't work,
    # however.
    #
    # (3,3) -> max_axis = 2
    max_axis = arr.ndim
    assert -1 <= axis <= max_axis, "axis out of bound"
    sl = [slice(None)]*(max_axis + 1)
    # e.g: [:,:,np.newaxis,...]
    sl[axis] = None
    return np.repeat(arr[sl], nstep, axis=axis)


def sum(arr, axis=None, keepdims=False, **kwds):
    """This numpy.sum() with some features implemented which can be found in
    numpy 2.0 (we have 1.5.1), probably a lot faster there, namely axis=tuple
    possible, keepdims keyword. Docstrings shamelessly stolen from numpy and
    adapted here and there.
    
    Parameters
    ----------
    arr : nd array
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default (`axis` =
        `None`) is to perform a sum over all the dimensions of the input array.
        `axis` may be negative, in which case it counts from the last to the
        first axis.
        If this is a tuple of ints, a sum is performed on multiple
        axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
        If this is set to True, the axes from ``axis`` are left in the result
        as dimensions with size one, and the reduction (sum) is performed for
        all remaining axes.
    **kwds : passed to np.sum().        

    Examples
    --------
    >>> a=rand(2,3,4)
    >>> num.sum(a)
    12.073636268676152
    >>> a.sum()
    12.073636268676152
    >>> num.sum(a, axis=1).shape
    (2, 4)
    >>> num.sum(a, axis=(1,)).shape
    (2, 4)
    >>> num.sum(a, axis=(0,2), keepdims=True).shape
    (2, 4)
    >>> num.sum(a, axis=(1,)) - num.sum(a, axis=1)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> num.sum(a, axis=(0,2)).shape
    (3,)
    >>> num.sum(a, axis=(0,2)) - a.sum(axis=0).sum(axis=1)
    array([ 0.,  0.,  0.])
    """
    # Recursion rocks!
    def _sum(arr, tosum):
        if len(tosum) > 0:
            # Choose axis to sum over, remove from list w/ remaining axes.
            axis = tosum.pop(0)
            _arr = arr.sum(axis=axis)
            # arr has one dim less now. Rename remaining axes accordingly.
            _tosum = [xx-1 if xx > axis else xx for xx in tosum]
            return _sum(_arr, _tosum)
        else:
            return arr
    
    axis_is_int = isinstance(axis, types.IntType)
    if (axis is None):
        if keepdims:
            raise StandardError("axis=None + keepdims=True makes no sense")
        else:
            return np.sum(arr, axis=axis, **kwds)
    elif axis_is_int and not keepdims:
        return np.sum(arr, axis=axis, **kwds)
    else:
        if axis_is_int:
            tosum = [axis]
        elif isinstance(axis, types.TupleType) or \
            isinstance(axis, types.ListType):
            tosum = list(axis)
        else:
            raise StandardError("illegal type for axis: %s" %str(type(axis)))
        if keepdims:
            alldims = range(arr.ndim)
            tosum = [xx for xx in alldims if xx not in tosum]
        return _sum(arr, tosum)


class Interpol2D(object):
    """Common 2D interpolator API. 
    
    This is for easy testing of multiple interpolators on a surface z = f(x,y),
    which is given as an unordered set of points."""
    def __init__(self, points=None, values=None, xx=None, yy=None,  dd=None,
                 what='rbf_multi', **initkwds):
        """
        Parameters
        ----------
        points : (npoints, 2)
        values : (npoints, 1)
        xx,yy : (npoints, 1)
            Use either `points` + `values` or `xx` + `yy` + `values` or `dd`.
        dd : pwtools.mpl.Data3D instance
        what : str, optional
            which interpolator to use
            'rbf_multi' : RBFN w/ multiquadric rbf
            'rbf_gauss' : RBFN w/ gaussian rbf
            'ct'        : scipy.interpolate.CloughTocher2DInterpolator
            'bispl'     : scipy.interpolate.bispl{rep,ev}    
        **initkwds : keywords passed on to the interpolator's constructor             

        possible keywords (examples):
        -----------------------------
        rbf :
            param='est'
            param=0.05
        ct :
            tol = 1e-6
        bispl :
            s = 1e-4 
            kx=3,ky=3 (default actually)
            nxest, nyest
        
        Examples
        --------
        >>> x=linspace(-5,5,20) 
        >>> y=x 
        >>> X,Y=np.meshgrid(x,y); X=X.T; Y=Y.T 
        >>> Z=(X+3)**2+(Y+4)**2 + 5 
        >>> dd=mpl.Data3D(X=X,Y=Y,Z=Z)
        >>> inter=num.Interpol2D(dd=dd, what='rbf_multi'); inter([[-3,-4],[0,0]])
        array([  5.0000001 ,  29.99999975])
        >>> inter=num.Interpol2D(dd=dd, what='rbf_gauss'); inter([[-3,-4],[0,0]])
        array([  5.00000549,  29.99999717])
        >>> inter=num.Interpol2D(dd=dd, what='ct'); inter([[-3,-4],[0,0]])
        array([  4.99762256,  30.010856  ])
        >>> inter=num.Interpol2D(dd=dd, what='bispl'); inter([[-3,-4],[0,0]])
        array([  5.,  30.])
        """
        if dd is None:
            if [xx, yy] == [None]*2:
                self.xx = points[:,0]
                self.yy = points[:,1]
                self.points = points
            elif points is None:
                self.xx = xx
                self.yy = yy
                self.points = np.array([xx,yy]).T
            else:
                raise StandardError("use points+values or xx+yy+values as input")
            self.values = values
        else:
            self.xx, self.yy, self.values, self.points = dd.xx, dd.yy, dd.zz, dd.XY

        from pwtools import rbf
        if what == 'rbf_multi':
            self.inter = rbf.RBFInt(self.points, self.values, rbf=rbf.RBFMultiquadric())
            self.inter.train('linalg', **initkwds)
            self.call = self.inter
        elif what == 'rbf_gauss':
            self.inter = rbf.RBFInt(self.points, self.values, rbf=rbf.RBFGauss())
            self.inter.train('linalg', **initkwds)
            self.call = self.inter
        elif what == 'ct':
            # Fail only when 'ct' is used. Don't do imports here, may be slow.
            if CloughTocher2DInterpolator is None:
                raise ImportError("could not import "
                    "scipy.interpolate.CloughTocher2DInterpolator")
            else:                    
                self.inter = CloughTocher2DInterpolator(self.points, self.values, **initkwds)
                self.call = self.inter
        elif what == 'bispl':
            nx = min(len(np.unique(self.xx)), int(sqrt(len(self.xx))))
            ny = min(len(np.unique(self.yy)), int(sqrt(len(self.yy))))
            _initkwds = {'kx': 3, 'ky': 3, 'nxest': 10*nx, 'nyest': 10*ny}
            _initkwds.update(initkwds)
            bispl = bisplrep(self.xx, self.yy, self.values, **_initkwds)
            def _call(points, bispl=bispl, **callkwds):
                # For unordered points, we need to loop.
                ret = [bisplev(points[ii,0], points[ii,1], bispl, **callkwds) for
                    ii in range(points.shape[0])]
                return np.array(ret)
            self.inter = _call
            self.call = _call                
   
    def __call__(self, points, **callkwds):
        """
        Parameters
        ----------
        points: 2d (M,2) or 1d (N,)
            M points in 2-dim space where to evalutae the interpolator
            (only one in 1d case)
        **callkwds : keywords passed to the interpolator's __call__ method            
        
        Returns
        -------
        Y : 1d array (M,)
            interpolated values
        """            
        points = np.asarray(points)
        if len(points.shape) == 1:
            points = points[None,:]
        return self.call(points, **callkwds)
    
    def get_min(self, x0=None):
        """Return [x,y] where z(x,y) = min(z) by minimizing z(x,y) w/
        scipy.optimize.fmin().
        
        Parameters
        ----------
        x0 : sequence, length (2,), optional
            Initial guess. If None then use the data grid point with the
            smallest `z` value.
        
        Returns
        -------
        [xmin, ymin]: 1d array (2,)
        """
        if x0 is None:
            idx0 = self.values.argmin()
            x0 = [self.xx[idx0], self.yy[idx0]]
        xopt = optimize.fmin(self, x0, disp=1, xtol=1e-8, ftol=1e-8, 
                    maxfun=1e4, maxiter=1e4)
        return xopt                        

def fempty(shape, dtype=np.float):
    return np.empty(shape, dtype=dtype, order='F')


def distsq(arrx, arry):
    """Squared distances between all points in `arrx` and `arry`:
        
        r_ij**2 = sum_k (arrx[i,k] - arry[j,k])**2.0
        i = 1..Mx
        j = 1..My
        k = 1..N

    This is like 
        scipy.spatial.distance.cdist(arrx, arry)**2.0
    
    This is a wrapper for _flib.distsq().

    Parameters
    ----------
    arrx, arry : ndarray (Mx,N), (My,N)
        Mx (My) points in N-dim space
    
    Returns
    -------
    2d array (Mx,My)
    """        
    nx, ny = arrx.shape[0], arry.shape[0]
    ndim = arrx.shape[1]
    ndimx, ndimy = arrx.shape[1], arry.shape[1]
    assert ndimx == ndimy, ("ndimx (%s, shape: %s) != ndimy (%s, shape: %s)" \
                           %(str(ndimx), 
                             str(arrx.shape), 
                             str(ndimy),
                             str(arry.shape)))
    # Allocating in F-order is essential for speed! For many points, this step
    # is actually the bottleneck, NOT the Fortran code! This is b/c if `dist`
    # is order='C' (numpy default), then the f2py wrapper makes a copy of the
    # array before starting to crunch numbers.
    dist = np.empty((nx, ny), dtype=arrx.dtype, order='F')
    return _flib.distsq(arrx, arry, dist, nx, ny, ndim)    



class DataND(object):
    """
    Transform 2d array `a2` to nd array `an`. The 2d array's last column are
    values on a grid represented by the nd array. The 2d array is the
    "flattened" version of the nd array. Works only for ordered axes where `a2`
    was generated by a nested loop over ordered 1d sequences, i.e.
    
    >>> nx,ny,nz = len(x),len(y),len(z)
    >>> for ii in range(nx):
    ...     for jj in range(ny):
    ...         for kk in range(nz):
    ...             idx = ii*ny*nz + jj*nz + kk 
    ...             a2[idx,0] = x[ii]
    ...             a2[idx,1] = y[jj]
    ...             a2[idx,2] = z[kk]
    ...             a2[idx,3] = <some value>
    >>> axes = [x,y,z]
    
    The `axes` are also extracted by numpy.unique() from `a2`'s columns,
    therefore only ordered axes work.

    The reverse operation `an` -> `a2` is not implemented ATM.

    Examples
    --------
    >>> from pwtools import num
    >>> # something to create grid values
    >>> a=iter(arange(1,100))
    >>> # Nested loop
    >>> a2=array([[x,y,z,a.next()] for x in [0,1,2] for y in [0,1] for z in [0,1,2,3]])  
    >>> nd=num.DataND(a2=a2)
    >>> nd.an.shape
    (3,2,4)
    >>> # nd array an[ii,jj,kk]
    >>> nd.an
    array([[[ 1,  2,  3,  4],
            [ 5,  6,  7,  8]],
           [[ 9, 10, 11, 12],
            [13, 14, 15, 16]],
           [[17, 18, 19, 20],
            [21, 22, 23, 24]]])
    >>> nd.axes
    [array([0, 1, 2]), array([0, 1]), array([0, 1, 2, 3])]
    """
    
    def __init__(self, a2=None, an=None, axes=None):
        """
        Parameters
        ----------
        arr : 2d array (nrows, ncols)
        
        Attributes
        -------
        nd : nd arry
        axes : list of 1d arrays
            The axes of the grid from np.unique()'ed ``ncols-1`` columns.
        """
        if an is None:
            self.a2 = a2
            self.an, self.axes = self.a2_to_an()

    def a2_to_an(self):
        axes = []
        dims = []
        for colidx in range(self.a2.shape[1]-1):
            a = np.unique(self.a2[:,colidx])
            axes.append(a)
            dims.append(len(a))
        assert np.product(dims) == self.a2.shape[0]
        idx = itertools.product(*tuple(map(range, dims)))
        an = np.empty(dims, dtype=self.a2.dtype)
        # an[1,2,3] == an[(1,2,3)], need way to eliminate loop over index array
        for ii,_idx in enumerate(idx):
            an[tuple(_idx)] = self.a2[ii,-1]
        return an, axes    


def rms(arr, nitems='all'):
    """RMS of all elements in a ndarray.
    
    Parameters
    ----------
    arr : ndarray
    nitems : {'all', float)
        normalization constant, the sum of squares is divided by this number,
        set to unity for no normalization, if 'all' then use nitems = number of
        elements in the array

    Returns
    -------
    rms : scalar
    """
    if nitems == 'all':
        nitems = float(arr.nbytes / arr.itemsize)
    else:
        nitems = float(nitems)
    rms = np.sqrt((arr**2.0).sum() / nitems)
    return rms        


def rms3d(arr, axis=0, nitems='all'):
    """RMS of 3d array along `axis`. Sum all elements of all axes != axis.
    
    Parameters
    ----------
    arr : 3d array
    axis : int
        The axis along which the RMS of all sub-arrays is to be computed
        (usually time axis in MD).
    nitems : {'all', float)
        normalization constant, the sum of squares is divided by this number,
        set to unity for no normalization, if 'all' then use nitems = number of
        elements in each sub-array along `axis`
    
    Returns
    -------
    rms : 1d array, (arr.shape[axis],)
    """
    # We could use num.sum() and would be able to generalize to nd arrays. But
    # not needed now.
    assert -1 <= axis <= 2, "allowed axis values: -1,0,1,2"
    assert arr.ndim == 3, "arr must be 3d array"
    if axis == -1:
        axis = arr.ndim - 1
    if nitems == 'all':
        sl = [slice(None)]*arr.ndim
        sl[axis] = 0 # pick out 1st sub-array along axis
        nitems = float(arr[sl].nbytes / arr.itemsize)
    else:
        nitems = float(nitems)
    if axis == 0:
        rms =  np.sqrt((arr**2.0).sum(1).sum(1) / nitems)
    elif axis == 1:
        rms =  np.sqrt((arr**2.0).sum(0).sum(1) / nitems)
    elif axis == 2:
        rms =  np.sqrt((arr**2.0).sum(0).sum(0) / nitems)
    return rms        
    

def poly_str(ndim, deg):
    """String representation of a `ndim`-poly of degree `deg`."""
    st = ''
    for ii,pwr in enumerate(poly_powers(ndim,deg)):
        xx = '*'.join('x%i^%i'%(i,n) for i,n in enumerate(pwr))
        term = 'a%i*' %ii + xx
        if st == '':
            st = term
        else:    
            st += ' + %s' %term
    return st        


def poly_powers(ndim, deg):
    """Powers for building a n-dim polynomial and columns of the n-dim
    Vandermonde matrix.

    Parameters
    ----------
    ndim : number of dimensions of the poly (e.g. 2 for f(x1,x2))
    deg : degree of the poly

    Returns
    -------
    powers : 2d array ``((deg+1)**ndim, ndim)``

    Examples
    --------
    For one dim, we have data points (x_i,y_i) and the to-be-fitted poly of order
    k is::
        
        f(x) = a0*x^0 + a1*x^1 + a2*x^2 + ... + ak*x^k

    The Vandermonde matrix A consists of all powers of x (cols) for all data
    points (rows) and each row has the form of the poly::

        [[x_0^0 x_0^1 ... x_0^k],
         [x_1^0 x_1^1 ... x_1^k],   
         ...
         [x_n^0 x_n^1 ... x_n^k]]
    
    To fit, we solve A . a = y, where a = [a0,...,ak].

    The retutned array `powers` has k rows, where each row holds the powers for
    one term in the poly. For ndim=1 and poly order k, we have::
        
        [[0],
         [1],
         ...
         [k]]
    
    and::

        [0] -> x^0
        [1] -> x^1
        ...
        [k] -> x^k
    
    Now, suppose we have 2 dims, thus data points (x0_i,x1_i,y_i) and a poly
    of order 2::
        
        f(x0,x1) = a0*x0^0*x1^0 + a1*x0^0*x1^1 + a2*x0^0*x1^2 + a3*x0^1*x1^0 +
                   a4*x0^1*x1^1 + a5*x0^1*x1^2 + a6*x0^2*x1^0 + a7*x0^2*x1^1 +
                   a8*x0^2*x1^2
    
    with 9 coeffs a = [a0,...,a8]. Therefore, ``powers.shape = (9,2)``::

        [[0, 0],
         [0, 1],
         [0, 2],
         [1, 0],
         [1, 1],
         [1, 2],
         [2, 0],
         [2, 1],
         [2, 2]]
    
    and:: 
        
        [0,0] -> x0^0*x1^0
        [1,2] -> x0^1*x1^2
        ...
    """
    return np.array([x for x in itertools.product(range(deg+1), 
                                                  repeat=ndim)])


def vander(points, deg):
    """N-dim Vandermonde matrix for data `points` and a polynom of degree
    `deg`.

    Parameters
    ----------
    points : see polyfit()
    deg : int
        Degree of the poly (e.g. 3 for cubic).
    
    Returns
    -------
    vander : 2d array (npoints, (deg+1)**ndim)
    """
    powers = poly_powers(points.shape[1], deg)
    # low memory version, slower
    ##npoints = points.shape[0]
    ##vand = np.empty((npoints, (deg+1)**ndim), dtype=float)
    ##for ipoint in range(npoints):
    ##    vand[ipoint,:] = (points[ipoint]**powers).prod(axis=1)
    tmp = (points[...,None] ** np.swapaxes(powers, 0, 1)[None,...])
    return tmp.prod(axis=1)


def polyfit(points, values, deg):
    """Fit nd polynomial of dregree `deg`. The dimension is ``points.shape[1]``.

    Parameters
    ----------
    points : nd array (npoints,ndim)
        `npoints` points in `ndim`-space, to be fitted by a `ndim` polynomial
        f(x0,x1,...,x{ndim-1}).
    values : 1d array        
    deg : int
        Degree of the poly (e.g. 3 for cubic).
    
    Returns
    -------
    fit : dict
        {coeffs, deg} where coeffs = 1d array ((deg+1)**ndim,) with poly
        coefficients. Input for polyval().
    """
    assert points.ndim == 2, "points must be 2d array"
    assert values.ndim == 1, "values must be 1d array"
    vand = vander(points, deg)
    return {'coeffs': np.linalg.lstsq(vand, values)[0], 'deg': deg}


def polyval(fit, points, der=0):
    """Evaluate polynomial generated by ``polyfit()`` on `points`.

    Parameters
    ----------
    fit, points : see polyfit()
    der : int, optional
        Derivative order. Only for 1D, uses np.polyder().
    
    Notes
    -----
    For 1D we provide "analytic" derivatives using np.polyder(). For ND, we
    didn't implement an equivalent machinery. For 2D, you might get away with
    fitting a bispline (see Interpol2D) and use it's derivs. For ND, try rbf.py's RBF
    interpolator which has at least 1st derivatives for arbitrary dimensions.
    """
    if der > 0:
        assert points.shape[1] == 1, "deriv only for 1d poly"
        dcoeffs = np.polyder(fit['coeffs'][::-1], m=der)
        return np.polyval(dcoeffs, points[:,0])
    else:        
        vand = vander(points, fit['deg'])
        return np.dot(vand, fit['coeffs'])


def inner_points_mask(points):
    """Mask array into `points` where ``points[msk]`` are all "inner" points,
    i.e. `points` with one level of edge points removed. For 1D, this is simply
    points[1:-1,:] (assuming ordered points). For ND, we calculate and remove
    the convex hull.

    Parameters
    ----------
    points : nd array (npoints, ndim)

    Returns
    -------
    msk : (npoints, ndim)
        Bool array.
    """
    msk = np.ones((points.shape[0],), dtype=bool)
    if points.shape[1] == 1:
        assert (np.diff(points[:,0]) >= 0.0).all(), ("points not monotonic")
        msk[0] = False
        msk[-1] = False
    else:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        edge_idx = np.unique(tri.convex_hull)
        msk.put(edge_idx, False)
    return msk    


def avgpolyfit(points, values, levels=1, degmin=1, degmax=1, degrange=None):
    """Same as polyfit(), but fit multiple polys. There are two types of fits,
    which can be combined. (1) Fit `levels` polys and remove one level of
    endpoints in each go from `points`. (2) Fit polys with varying degrees (see
    `degmin` + `degmax` or `degrange`).

    Parameters
    ----------
    points, values : see polyfit()
    levels : int
        Fit this many polys and remove endpoints from `points` for levels >= 1.
    degmin, degmax : int
        min and max degree
    degrange : sequence
        range of poly degrees, use either this or `degmin` + `degmax`
    
    Returns
    -------
    fit : dict
        {fits, errors, degrees, npoints}, `fits` = sequence of dicts from
        polyfit(), `errors` =  1d array (ndeg*levels,) with RMS errors of each
        fit, `degrees` = 1d array (ndeg*levels,) with poly degrees, npoints =
        1d array (ndeg*levels,) with used number of points.
    
    Examples
    --------
    >>> fit=avgpolyfit(points, values, degmin=3, degmax=7, levels=5)
    >>> avgpolyval(fit, newpoints)
    >>>
    >>> # the same as polyfit(points, values, deg=3)
    >>> fit=avgpolyfit(points, values, degmin=3, degmax=3, levels=1)
    >>> fit=avgpolyfit(points, values, degrange=[3], levels=1)
    """
    # In [1]_ (see avgpolyval) they say they use both multiple poly dregees and
    # end point removal. But multiple degrees is actually pretty useless since
    # a poly of degree `deg` contains all terms from a `deg`-1 poly. E.g. If we
    # fit x**2 data with a deg=2 poly, we get the same results as with an
    # deg=10 poly: all coeffs for deg != 2 are zero. Maybe we need to have a
    # look at the GIBBS code again -- sparsely commented F77.
    ndim = points.shape[1]
    if degrange is None:
        assert degmin <= degmax, "degmin (%i) > degmax (%i)" %(degmin, degmax)
        degs = range(degmin, degmax+1)
    else:
        degs = degrange
    ndeg = len(degs)
    errors = []
    fits = []
    degrees = []
    npoints = []
    for ilev in range(levels):
        if ilev >= 1:
            msk = inner_points_mask(points)
            points = points[msk,...]
            values = values[msk]
        for deg in degs:
            fit = polyfit(points, values, deg)
            err = rms(polyval(fit, points) - values, nitems=points.shape[0])
            fits.append(fit)
            errors.append(err)
            degrees.append(deg)
            npoints.append(points.shape[0])
    return {'fits': fits, 'errors': np.array(errors), 'degrees': np.array(degrees, dtype=float),
            'npoints': np.array(npoints, dtype=float)}


def avgpolyval(fit, points, der=0, weight_type=1):
    """Use poly fits from avgpolyfit() and average them according to their RMS
    error.
    
    Parameters
    ----------
    fit : tuple
        Result of avgpolyfit().
    points : see polyfit()
    der : see polyval()
    weight_type : int
        1 : without poly degree
        2 : with poly degree as in [1]_

    Notes
    -----
    The weighting and averaging is due to [1]_, but by default we don't include
    the polynomial's degree in the weight as this produces worse fits
    sometimes.
    
    .. [1] Blanco, M.; Francisco, E. & Luana, V., 
       "GIBBS: isothermal-isobaric thermodynamics of solids from energy
       curves using a quasi-harmonic Debye model", Computer Physics
       Communications, 2004, 158, 57-72
    """
    fits, errors, degrees, npoints = \
        fit['fits'], fit['errors'], fit['degrees'], fit['npoints']
    if weight_type == 1:
        err = errors / errors.min() 
    elif weight_type == 2:
        err = errors / errors.min() * \
              (degrees + 1.0) / npoints
    else:
        raise StandardError("unknown weight_type")
    weights = np.exp(-err**2.0)
    weights = weights / weights.sum()
    ret = np.zeros((points.shape[0],), dtype=float)
    for ft,wght in itertools.izip(fits, weights):
        wft = copy.deepcopy(ft)
        wft['coeffs'] *= wght
        ret += polyval(wft, points, der=der)
    return ret        


class PolyFit(object):
    """High level interface to [avg]poly{fit,val}, similar to Spline and
    Interpol2D. 
    __init__: `points` must be (npoints,ndim) even if ndim=1.
    __call__: `points` can be (ndim,) instead of (1,ndim), need this if called in
               fmin()
    """
    def __init__(self, points, values, **kwds):
        """
        Parameters
        ----------
        points : nd array (npoints, ndim)
        values : 1d array (npoints,)
        **kwds : keywords to *polyfit()
        """
        self.points = self._fix_shape_init(points)
        assert self.points.ndim == 2, "points is not 2d array"
        self.values = values
        if self._has_keys(kwds, ['degrange', 'degmin', 'degmax', 'levels']):
            if kwds.has_key('deg'):
                deg = kwds['deg']
                kwds.pop('deg')
                kwds['degrange'] = [deg]
            self.fitfunc = avgpolyfit
            self.evalfunc = avgpolyval
        else:
            self.fitfunc = polyfit
            self.evalfunc = polyval
        self.fit = self.fitfunc(self.points, self.values, **kwds)
    
    @staticmethod
    def _has_keys(dct, keys):
        if type(keys) != type([]):
            keys = [keys]
        ret = False            
        for k in keys:
            if dct.has_key(k):
                ret = True
                break
        return ret                
    
    @staticmethod
    def _fix_shape_init(points):
        return points
    
    @staticmethod
    def _fix_shape_call(points):
        if points.ndim == 1:
            return points[None,:]
        else:
            return points
        
    def __call__(self, points, **kwds):
        points = self._fix_shape_call(points)
        ret = self.evalfunc(self.fit, points, **kwds)
        if points.shape[0] == 1:
            return ret[0]
        else:
            return ret

    def get_min(self, x0=None, **kwds):
        _kwds = dict(disp=1, xtol=1e-12, ftol=1e-8, maxfun=1e4, maxiter=1e4)
        _kwds.update(kwds)
        if x0 is None:
            idx = self.values.argmin()
            x0 = self.points[idx,...]
        xopt = optimize.fmin(self, x0, **_kwds)
        return xopt                        


class PolyFit1D(PolyFit):
    """1D special case version of PolyFit which handles 1d and scalar `points`
    Also `get_min()` uses the root of the poly's 1st derivative instead of
    fmin().

    __init__: points (npoints,1) or (npoints,)
    __call__: points (npoints,1) or (npoints,) or scalar
    """
    def __init__(self, *args, **kwds):
        """
        Parameters
        ----------
        points : nd array (npoints,) or (npoints,1) 
        values : 1d array (npoints,)
        **kwds : keywords to *polyfit()
        """
        super(PolyFit1D, self).__init__(*args, **kwds)
        assert self.points.ndim == 2 and self.points.shape[1] == 1, \
            ("points has wrong shape: %s, expect (npoints,1)" \
            %str(self.points.shape))
    
    @staticmethod
    def _fix_shape_init(points):
        pp = np.asarray(points)
        # 1 -> (1,1)            
        if pp.ndim == 0:
            return np.array([[pp]])
        # (N,) -> (N,1)
        elif pp.ndim == 1:            
            return pp[:,None]
        elif  pp.ndim == 2:
            return pp
        else:
            raise ValueError("wrong shape or dim")
    
    _fix_shape_call = _fix_shape_init

    def _findroot(self, func, x0=None, xab=None, **kwds):
        if x0 is not None:
            xx = optimize.newton(func, x0, **kwds)
        else:
            if xab is None:
                xab = [self.points[0,0], self.points[-1,0]]
            xx = optimize.brentq(func, xab[0], xab[1], **kwds)
        return xx    
    
    def get_min(self, x0=None, xab=None, **kwds):
        return self._findroot(lambda x: self(x, der=1), x0=x0, xab=xab, 
                              **kwds)
                      
