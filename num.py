# num.py : numpy/scipy like stuff.

import os
import numpy as np
from scipy.optimize import brentq, newton
from scipy.interpolate import splev, splrep

def normalize(a):
    """Normalize array by it's max value. Works also for complex arrays.

    example:
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
    
    args:
    -----
    a, b : 1d arrays
    num : int

    returns:
    --------
    array (num, len(a)), each row is a "point" between `a` and `b`
    """
    assert a.ndim == b.ndim == 1, "expect 1d arrays"
    assert len(a) == len(b), "`a` and `b` must have equal length"
    # distance vec connecting a and b
    dv = b-a
    if endpoint:
        ddv = dv/float(num-1)
    else:        
        ddv = dv/float(num)
    ret = np.empty((num, len(dv)), dtype=float)
    ret[...] = ddv
    ret[0,:] = a
    return np.cumsum(ret, axis=0)
    
def norm_int(y, x, area=1.0):
    """Normalize integral area of y(x) to `area`.
    
    args:
    -----
    x,y : numpy 1d arrays
    area : float

    returns:
    --------
    scaled y

    notes:
    ------
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


def deriv_fd(y, x=None, n=1):
    """n-th derivative for 1d arrays of possibly nonuniformly sampled data.
    Returns matching x-axis for plotting. Simple finite differences are used:
    f'(x) = [f(x+h) - f(x)] / h
    
    args:
    -----
    x,y : 1d arrays of same length
        if x=None, then x=arange(len(y)) is used
    n : int
        order of the derivative
    
    returns:
    --------
    xd, yd
    xd : 1d array, (len(x)-n,)
        matching x-axis
    yd : 1d array, (len(x)-n,)
        n-th derivative of y at points xd

    notes:
    ------
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

    example:
    --------
    >>> x=sort(rand(100)*10); y=sin(x); plot(x,y, 'o-'); plot(x,cos(x), 'o-')
    >>> x1,y1=deriv_fd(y,x,1) # cos(x)
    >>> x2,y2=deriv_fd(y,x,2) # -sin(x)
    >>> plot(x1, y1, lw=2) # cos(x)
    >>> plot(x2, -y2, lw=2) # sin(x)
    >>> x=linspace(0,10,100); y=sin(x); plot(x,y, 'o-'); plot(x,cos(x), 'o-')
    >>> ...
    
    see also:
    ---------
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
    
    args:
    -----
    x,y : 1d arrays of same length
        if x=None, then x=arange(len(y)) is used
    xnew : {None, 1d array)
        x-axis to evaluate the derivative, if None then xnew=x
    n : int
        order of the derivative, can only be <= k 
    fullout : bool
        return xd, yd or just yd
    splrep_kwargs : keyword args to scipy.interpolate.splrep, default: k=3, s=0

    returns:
    --------
    if fullout:
        xd, yd
    else:
        yd
    xd : 1d array, (len(x) or len(xnew),)
    yd : 1d array, (len(x) or len(xnew),)
        n-th derivative of y at points xd
    
    notes:
    ------
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
    x0 = brentq(func, x[0], x[-1])
    return np.array([x0, splev(x0, tck)])


def findmin(x, y):
    """Find minimum of x-y curve by searching for the root of the 1st
    derivative of a spline thru x,y. `x` must be sorted min -> max and the
    interval [x[0], x[-1]] must contain the minimum.
    
    This is intended for quick interactive work. For working with
    pre-calculated splines, see Spline.

    args:
    -----
    x,y : 1d arrays

    returns:
    --------
    array([x0, y(x0)])
    """
    return _splroot(x, y, der=1)


def findroot(x, y):
    """Find root of x-y curve by searching for the root of a spline thru x,y.
    `x` must be sorted min -> max and the interval [x[0], x[-1]] must contain
    the root.

    This is intended for quick interactive work. For working with
    pre-calculated splines, see Spline.
    
    args:
    -----
    x,y : 1d arrays

    returns:
    --------
    array([x0, y(x0)])
    """
    return _splroot(x, y, der=0)


class Spline(object):
    """Wrapper around scipy.interpolate.splrep/splev with some nice features
    like y->x lookup and interpolation accuracy check etc. It basically
    simplifies setting up a spline interpolation and holds x-y data plus the
    spline knots (self.tck) together in one place. You can work with the
    methods here, but you can also use the normal tck (self.tck) in
    scipy.interpolation.splev() etc.

    example:
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
        args:
        -----
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
        for key, val in {'s':0, 'k':3}.iteritems():
            if not splrep_kwargs.has_key(key):
                splrep_kwargs[key] = val
        self.splrep_kwargs = splrep_kwargs
        self.tck = splrep(self.x, self.y, **splrep_kwargs)
        if checkeps:
            err = np.abs(self.splev(self.x) - self.y)
            assert (err < self.eps).all(), \
                    ("spline not accurate to eps=%e, max(error)=%e, raise eps"\
                    %(self.eps, err.max()))

    def __call__(self, *args, **kwargs):
        return self.splev(*args, **kwargs)

    def _findroot(self, func, x0=None, xab=None):
        if x0 is not None:
            xx = newton(func, x0)
        else:
            if xab is None:
                xab = [self.x[0], self.x[-1]]
            xx = brentq(func, xab[0], xab[1])
        return xx    
    
    def is_mono(self):
        """Return True if the curve described by the spline is monotonic."""
        tmp = np.diff(np.sign(np.diff(self.splev(self.x))))       
        return (tmp == 0).all()

    def splev(self, x, *args, **kwargs):
        return splev(x, self.tck, *args, **kwargs)

    def invsplev(self, y0, x0=None, xab=None):
        """Lookup x for a given y, i.e. "inverse spline evaluation", hence
        the same. Find xx where y(xx) == y0 by calculating the root of y(x) -
        y0. We can use Newton's (x0) or Brent's (xab) methods. Use only one of
        them. If neither is given, we use xab=[x[0], x[-1]].
       
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
        
        args:
        -----
        x0 : float
            start guess for Newton secant method
        xab : length 2 sequence
            interval for Brent method
        
        returns:
        --------
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
        """Return xx where y(xx) = min(y) by calculating the root of the
        spline's 1st derivative.
        
        args:
        -----
        x0 or xab: see self.invsplev()
        
        returns:
        --------
        xx : scalar
            min(y) = y(xx)
        """
        func = lambda x: self.splev(x, der=1)
        return self._findroot(func, x0=x0, xab=xab)
    
    def get_root(self, x0=None, xab=None):
        """Return xx where y(xx) = 0 by calculating the root of the spline.
        This function is actually redundant b/c it can be done with
        self.invsplev(0.0, ...), i.e. lookup x where y=0, which is exactly the
        root. But we keep it for reference and convenience.
        
        args:
        -----
        x0 or xab: see self.invsplev()
        
        returns:
        --------
        xx : scalar
            y(xx) = 0
        """
        return self._findroot(self.splev, x0=x0, xab=xab)
    

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


