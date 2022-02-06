import os, copy
import types
import itertools
import numpy as np
from math import sqrt, sin, cos, radians, pi
import scipy.optimize as optimize
from scipy.interpolate import bisplrep, \
    bisplev, splev, splrep
from scipy.integrate import simps, trapz
from pwtools import _flib
import warnings
##warnings.simplefilter('always')

# Hack for older scipy versions.
try:
    from scipy.interpolate import CloughTocher2DInterpolator, \
        NearestNDInterpolator, LinearNDInterpolator
except ImportError:
    # Don't throw a warning here b/c (1) this module is imported often and that
    # would annoy anyone to no end and (2) this interpolator isn't used much,
    # only for experimentation. It's enough to fail inside Interpol2D() if
    # needed.
    CloughTocher2DInterpolator = None
    NearestNDInterpolator = None
    LinearNDInterpolator = None

# constants
EPS = np.finfo(float).eps

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


def norm_int(y, x, area=1.0, scale=True, func=simps):
    """Normalize integral area of y(x) to `area`.

    Parameters
    ----------
    x,y : numpy 1d arrays
    area : float
    scale : bool, optional
        Scale x and y to the same order of magnitude before integration.
        This may be necessary to avoid numerical trouble if x and y have very
        different scales.
    func : callable
        Function to do integration (like scipy.integrate.{simps,trapz,...}
        Called as ``func(y,x)``. Default: simps

    Returns
    -------
    scaled y

    Notes
    -----
    The argument order y,x might be confusing. x,y would be more natural but we
    stick to the order used in the scipy.integrate routines.
    """
    if scale:
        fx = np.abs(x).max()
        fy = np.abs(y).max()
        sx = x / fx
        sy = y / fy
    else:
        fx = fy = 1.0
        sx, sy = x, y
    # Area under unscaled y(x).
    _area = func(sy, sx) * fx * fy
    return y*area/_area


def deriv_spl(y, x=None, xnew=None, n=1, fullout=False, **splrep_kwargs):
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
    for key, val in {'s':0, 'k':3}.items():
        if key not in splrep_kwargs:
            splrep_kwargs[key] = val
    yd = splev(xnew, splrep(x, y, **splrep_kwargs), der=n)
    if fullout:
        return xnew, yd
    else:
        return yd


def findmin(x, y):
    """Find minimum of x-y curve by searching for the root of the 1st
    derivative of a spline thru x,y. `x` must be sorted min -> max and the
    interval [x[0], x[-1]] must contain the minimum.

    This is intended for quick interactive work. For working with
    pre-calculated splines, see :class:`Spline`.

    Parameters
    ----------
    x,y : 1d arrays

    Returns
    -------
    array([x0, y(x0)])
    """
    warnings.warn("use Spline(x,y).get_min()", DeprecationWarning)
    spl = Spline(x,y)
    x0 = spl.get_min()
    return np.array([x0, spl(x0)])


def findroot(x, y):
    """Find root of x-y curve by searching for the root of a spline thru x,y.
    `x` must be sorted min -> max and the interval [x[0], x[-1]] must contain
    the root.

    This is intended for quick interactive work. For working with
    pre-calculated splines, see :class:`Spline`.

    Parameters
    ----------
    x,y : 1d arrays

    Returns
    -------
    array([x0, y(x0)])
    """
    warnings.warn("use Spline(x,y).get_root()", DeprecationWarning)
    spl = Spline(x,y)
    x0 = spl.get_root()
    return np.array([x0, spl(x0)])


class Fit1D:
    """Base class for 1D data fit/interpolation classes (:class:`Spline`,
    :class:`PolyFit1D`). It provides :meth:`get_min`, :meth:`get_max`,
    :meth:`get_root`, :meth:`is_mono`.

    The assumed API is that the ``__call__`` method has a kwd `der` which
    causes it to calculate derivatives, i.e. ``__call__(x, der=1)`` is the
    first deriv."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # XXX once the now deprecated module-level findroot() is gone, we can turn
    # this into a new module-level findroot(), with another API however.
    def _findroot(self, func, x0=None, xab=None, **kwds):
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
        **kwds :
            passed to scipy root finder (newton() or brentq())

        Returns
        -------
        xx : scalar
            the root of func(x)
        """
        if x0 is not None:
            xx = optimize.newton(func, x0, **kwds)
        else:
            if xab is None:
                xab = [self.x[0], self.x[-1]]
            xx = optimize.brentq(func, xab[0], xab[1], **kwds)
        return xx

    def is_mono(self):
        """Return True if the curve described by the fit function f(x) is
        monotonic."""
        tmp = np.diff(np.sign(np.diff(self(self.x))))
        return (tmp == 0).all()

    def get_min(self, x0=None, xab=None, **kwds):
        """Return x where y(x) = min(y) by calculating the root of the
        fit's 1st derivative (by calling ``self(x, der=1)``).

        Parameters
        ----------
        x0 or xab :
            see :meth:`_findroot`
        **kwds :
            passed to :meth:`_findroot`

        Returns
        -------
        xx : scalar
            min(y) = y(xx)
        """
        return self._findroot(lambda x: self(x, der=1), x0=x0, xab=xab, **kwds)

    def get_max(self, x0=None, xab=None, **kwds):
        """Convenience method. Same as :meth:`get_min`, just for local maxima,
        simply using ``-self(x, der=1)``."""
        return self._findroot(lambda x: -self(x, der=1), x0=x0, xab=xab, **kwds)

    def get_root(self, x0=None, xab=None, **kwds):
        """Return x where y(x) = 0 by calculating the root of the fit function.

        In :class:`Spline`, this is the same as ``Spline.invsplev(0.0, ...)``,
        i.e. lookup x where y=0, which is exactly the root.

        Parameters
        ----------
        x0 or xab :
            see :meth:`_findroot`
        **kwds :
            passed to :meth:`_findroot`

        Returns
        -------
        xx : scalar
            y(xx) = 0
        """
        return self._findroot(self, x0=x0, xab=xab, **kwds)


class Spline(Fit1D):
    """:class:`Fit1D`-based spline interpolator.

    Like ``scipy.interpolate.UnivariateSpline``, this is a wrapper around
    ``scipy.interpolate.splrep/splev`` with some nice features like y->x lookup
    and interpolation accuracy check etc. It basically simplifies setting up a
    spline interpolation and holds x-y data plus the spline knots
    (``self.tck``) together in one place. You can work with the methods here,
    but you can also use the normal tck (``self.tck``) in
    ``scipy.interpolate.splev()`` etc.

    Examples
    --------
    >>> from scipy.interpolate import splev
    >>> from pwtools import num
    >>> x = linspace(0,10,100)
    >>> y = sin(x)
    >>> sp = num.Spline(x,y)
    >>> plot(x,y)
    >>> plot(x, sp(x))
    >>> plot(x, splev(x, sp.tck)) # the same
    >>> plot(x, sp(x, der=1), label='1st derivative')
    >>> xx = sp.invsplev(0.5, xab=[0, pi/2])
    >>> print("at %f degrees, sin(x) = 0.5" %(xx/pi*180))
    >>>
    >>> y = x**2 - 5
    >>> sp = num.Spline(x,y)
    >>> print("the root is at x=%f" %sp.invsplev(0.0))
    >>> legend()
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
        super(Spline, self).__init__(x,y)
        self.arr_zero_dim_t = type(np.array(1.0))
        self.eps = eps
        assert (np.diff(self.x) >= 0.0).all(), ("x wronly ordered")
        self.splrep_kwargs = {'s':0, 'k':3}
        self.splrep_kwargs.update(splrep_kwargs)
        self.tck = splrep(self.x, self.y, **self.splrep_kwargs)
        if checkeps:
            err = np.abs(self(self.x) - self.y)
            assert (err < self.eps).all(), \
                    ("spline not accurate to eps=%e, max(error)=%e, raise eps"\
                    %(self.eps, err.max()))

    def __call__(self, x, *args, **kwargs):
        ret = splev(x, self.tck, *args, **kwargs)
        # splev() retrns array(<number>) for scalar input, convert to scalar
        # float
        if type(ret) == self.arr_zero_dim_t and ret.ndim == 0:
            return float(ret)
        else:
            return ret

    def splev(self, x, *args, **kwds):
        warnings.warn("use Spline(x,y)(new_x) instead of Spline.splev()", DeprecationWarning)
        return self(x, *args, **kwds)

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
        func = lambda x: self(x) - y0
        return self._findroot(func, x0=x0, xab=xab)


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
    slices = tuple(slices)
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
    :func:`numpy.repeat`
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
    return np.repeat(arr[tuple(sl)], nstep, axis=axis)


def sum(arr, axis=None, keepdims=False, **kwds):
    """This numpy.sum() with some features implemented which can be found in
    numpy v1.7 and later: `axis` can be a tuple to select arbitrary axes to sum
    over.

    We also have a `keepdims` keyword, which however works completely different
    from numpy. Docstrings shamelessly stolen from numpy and adapted here
    and there.

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
        If this is set to True, the axes from `axis` are left in the result
        and the reduction (sum) is performed for all remaining axes. Therefore,
        it reverses the `axis` to be summed over.
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
    >>> # same as axis=1, i.e. it inverts the axis over which we sum
    >>> num.sum(a, axis=(0,2), keepdims=True).shape
    (2, 4)
    >>> # numpy's keepdims has another meaning: it leave the summed axis (0,2)
    >>> # as dimension of size 1 to allow broadcasting
    >>> numpy.sum(a, axis=(0,2), keepdims=True).shape
    (1, 3, 1)
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

    axis_is_int = isinstance(axis, int)
    if (axis is None):
        if keepdims:
            raise Exception("axis=None + keepdims=True makes no sense")
        else:
            return np.sum(arr, axis=axis, **kwds)
    elif axis_is_int and not keepdims:
        return np.sum(arr, axis=axis, **kwds)
    else:
        if axis_is_int:
            tosum = [axis]
        elif isinstance(axis, tuple) or \
            isinstance(axis, list):
            tosum = list(axis)
        else:
            raise Exception("illegal type for axis: %s" %str(type(axis)))
        if keepdims:
            alldims = range(arr.ndim)
            tosum = [xx for xx in alldims if xx not in tosum]
        return _sum(arr, tosum)


class Interpol2D:
    """Common 2D interpolator API.

    The API is the same as in ``scipy.interpolate``, e.g.

    >>> inter = scipy.interpolate.SomeInterpolatorClass(points, values)
    >>> new_values = inter(new_points)

    This is for easy testing of multiple interpolators on a surface z = f(x,y),
    which is given as an unordered set of points.
    """
    def __init__(self, points=None, values=None, xx=None, yy=None,  dd=None,
                 what='rbf_inv_multi', **initkwds):
        """
        Parameters
        ----------
        points : (npoints, 2)
        values : (npoints, 1)
        xx,yy : (npoints, 1)
            Use either `points` + `values` or `xx` + `yy` + `values` or `dd`.
        dd : pwtools.mpl.Data2D instance
        what : str, optional
            which interpolator to use

            | 'rbf_multi' : RBFN w/ multiquadric rbf, see :class:`~pwtools.rbf.Rbf`
            | 'rbf_inv_multi' : RBFN w/ inverse multiquadric rbf
            | 'rbf_gauss' : RBFN w/ gaussian rbf
            | 'poly'      : :class:`PolyFit`
            | 'bispl'     : scipy.interpolate.bispl{rep,ev}
            | 'ct'        : scipy.interpolate.CloughTocher2DInterpolator
            | 'linear'    : scipy.interpolate.LinearNDInterpolator
            | 'nearest'   : scipy.interpolate.NearestNDInterpolator
        **initkwds : keywords passed on to the interpolator's constructor or
            fit() method (RBF case)

        Notes
        -----
        Despite the name "Interpol2D", the RBF methods 'rbf_*' as well as 'poly' are
        actually fits (least squares regression). You can force interpolation with
        the RBF methods using the ``r=0`` keyword (see
        :meth:`pwtools.rbf.Rbf.fit`), which will use ``scipy.linalg.solve``
        without regularization.

        The methods 'ct', 'linear' and of course 'nearest' can be inaccurate
        (see also ``test/test_interpol.py``). Use only for plotting, not for
        data evaluation, i.e. accurate minimas etc.

        Except for 'bispl', all interpolators do actually work in ND as well, as
        does :meth:`get_min`.

        Possible keywords (examples):

        | rbf :
        |     p='mean' [,r=None] (default)    # linalg.lstsq
        |     p='scipy', r=1e-8               # linalg.solve w/ regularization
        |     p=3.5, r=0                      # linalg.solve w/o regularization
        | ct :
        |     tol = 1e-6 (default)
        | bispl :
        |     s = 1e-4
        |     kx = 3, ky = 3 (default)
        |     nxest, nyest
        | poly :
        |     deg = 5

        Examples
        --------
        >>> from pwtools import num, mpl
        >>> x=linspace(-5,5,20)
        >>> y=x
        >>> X,Y=np.meshgrid(x,y); X=X.T; Y=Y.T
        >>> Z=(X+3)**2+(Y+4)**2 + 5
        >>> dd=mpl.Data2D(X=X,Y=Y,Z=Z)
        >>> fmt="what: {:15} target: [5,30] result: {}"
        >>> for method in ['rbf_multi', 'rbf_inv_multi',
        ...                'rbf_gauss', ('poly', {'deg': 5}),
        ...                'ct', 'bispl', 'linear', 'nearest']:
        ...     if isinstance(method, tuple):
        ...         what = method[0]
        ...         kwds = method[1]
        ...     else:
        ...         what = method
        ...         kwds = {}
        ...     inter=num.Interpol2D(dd=dd, what=what, **kwds)
        ...     print(fmt.format(what, inter([[-3,-4],[0,0]])))
        what: rbf_multi       target: [5,30] result: [  5.00000005  29.99999959]
        what: rbf_inv_multi   target: [5,30] result: [  4.99999808  29.99999798]
        what: rbf_gauss       target: [5,30] result: [  5.00000051  30.00000352]
        what: poly            target: [5,30] result: [  5.  30.]
        what: ct              target: [5,30] result: [  4.99762256  30.010856  ]
        what: bispl           target: [5,30] result: [  5.  30.]
        what: linear          target: [5,30] result: [  5.06925208  30.13850416]
        what: nearest         target: [5,30] result: [  5.01385042  33.82271468]
        """
        if dd is None:
            if xx is None and yy is None:
                self.xx = points[:,0]
                self.yy = points[:,1]
                self.points = points
            elif points is None:
                self.xx = xx
                self.yy = yy
                self.points = np.array([xx,yy]).T
            else:
                raise Exception("use points+values or xx+yy+values as input")
            self.values = values
        else:
            self.xx, self.yy, self.values, self.points = dd.xx, dd.yy, dd.zz, dd.XY

        # need to import here b/c of circular dependency rbf.py <-> num.py
        from pwtools import rbf
        if what == 'rbf_multi':
            self.inter = rbf.Rbf(self.points, self.values,
                                 rbf='multi', **initkwds)
            self.call = self.inter
        elif what == 'rbf_inv_multi':
            self.inter = rbf.Rbf(self.points, self.values,
                                 rbf='inv_multi', **initkwds)
            self.call = self.inter
        elif what == 'rbf_gauss':
            self.inter = rbf.Rbf(self.points, self.values, rbf='gauss', **initkwds)
            self.call = self.inter
        elif what == 'poly':
            self.inter = PolyFit(self.points, self.values, scale=True, **initkwds)
            self.call = self.inter
            self.call = self._poly_format_return
        elif what == 'ct':
            if CloughTocher2DInterpolator is None:
                raise ImportError("could not import "
                    "scipy.interpolate.CloughTocher2DInterpolator")
            else:
                self.inter = CloughTocher2DInterpolator(self.points,
                                                        self.values,
                                                        **initkwds)
                self.call = self.inter
        elif what == 'nearest':
            if NearestNDInterpolator is None:
                raise ImportError("could not import "
                    "scipy.interpolate.NearestNDInterpolator")
            else:
                self.inter = NearestNDInterpolator(self.points, self.values,
                                               **initkwds)
                self.call = self.inter
        elif what == 'linear':
            if LinearNDInterpolator is None:
                raise ImportError("could not import "
                    "scipy.interpolate.LinearNDInterpolator")
            else:
                self.inter = LinearNDInterpolator(self.points, self.values,
                                                   **initkwds)
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
        else:
            raise Exception("unknown interpolator type: %s" %what)

    # See pwtools.test.test_polyfit.test_api: work around subtle PolyFit API
    # difference to all other interpolators w/o breaking neither Interpol2D's
    # nor PolyFit's API
    def _poly_format_return(self, *args, **kwds):
        return np.atleast_1d(self.inter(*args, **kwds))

    def __call__(self, points, **callkwds):
        """
        Parameters
        ----------
        points : 2d (M,2) or 1d (N,)
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

    def get_min(self, x0=None, **kwds):
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
        _kwds = dict(disp=0, xtol=1e-12, ftol=1e-8, maxfun=1e4, maxiter=1e4)
        _kwds.update(kwds)
        if x0 is None:
            idx0 = self.values.argmin()
            x0 = [self.xx[idx0], self.yy[idx0]]
        xopt = optimize.fmin(self, x0, **_kwds)
        return xopt

def fempty(shape, dtype=np.float64):
    return np.empty(shape, dtype=dtype, order='F')


def distsq(arrx, arry):
    r"""Squared distances between all points in `arrx` and `arry`:

    .. math::

        r_{ij}^2 = \sum_k (\texttt{arrx}[i,k] - \texttt{arry}[j,k])^2 \\
        i = 1..M_x \\
        j = 1..M_y \\
        k = 1..N

    This is like
        scipy.spatial.distance.cdist(arrx, arry)**2.0

    This is a wrapper for :func:`pwtools._flib.distsq`.

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



class DataND:
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
    nitems : {'all', float}
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
    nitems : {'all', float}
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
        nitems = float(arr[tuple(sl)].nbytes / arr.itemsize)
    else:
        nitems = float(nitems)
    if axis == 0:
        rms =  np.sqrt((arr**2.0).sum(1).sum(1) / nitems)
    elif axis == 1:
        rms =  np.sqrt((arr**2.0).sum(0).sum(1) / nitems)
    elif axis == 2:
        rms =  np.sqrt((arr**2.0).sum(0).sum(0) / nitems)
    return rms


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

    The returned array `powers` has k rows, where each row holds the powers for
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
    return np.array(list(itertools.product(range(deg+1), repeat=ndim)))


def vander(points, deg):
    """N-dim Vandermonde matrix for data `points` and a polynomial of degree
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


def polyfit(points, values, deg, scale=True, scale_vand=False):
    """Fit nd polynomial of dregree `deg`. The dimension is ``points.shape[1]``.

    Parameters
    ----------
    points : nd array (npoints,ndim)
        `npoints` points in `ndim`-space, to be fitted by a `ndim` polynomial
        f(x0,x1,...,x{ndim-1}).
    values : 1d array
    deg : int
        Degree of the poly (e.g. 3 for cubic).
    scale: bool, optional
        Scale `points` and `values` to unity internally before fitting.
        ``fit['coeffs']`` are for scaled data. ``polyval`` handles that
        transparently.
    scale_vand : bool, optional
        scale Vandermonde matrix as in numpy.polyfit (devide by column norms to
        improve condition number)

    Returns
    -------
    fit : dict
        {coeffs, deg, pscale, vscale, pmin, vmin} where coeffs = 1d array
        ((deg+1)**ndim,) with poly coefficients and `*min` and `*scale` are for
        data scaling. Input for polyval().

    Notes
    -----
    `scale`: `numpy.polyfit` does only `scale_vand` by default, which seems to be
    enough for most real world data. The new `np.polynomial.Polynomial.fit`
    now does the equivalent of what we do here with `scale`, **but they do it
    only for `points`, not `values`**. They map to [-1,1], we use [0,1].

    In most tests so far, `scale_vand` and `scale` have pretty much the same
    effect: enable fitting data with very different scales on x and y.

    Because ``fit['coeffs']`` are w.r.t. scaled data, you cannot compare them
    to the result of `np.polyfit` directly. Only with `scale=False` you can
    compare the coeffs, which should be the same up to numerical noise.
    However, you may simply compare the resulting fits, evaluated at the same
    points.

    See Also
    --------
    :class:`PolyFit`, :class:`PolyFit1D`, :func:`polyval`
    """
    assert points.ndim == 2, "points must be 2d array"
    assert values.ndim == 1, "values must be 1d array"
    assert len(values) == points.shape[0], (
        "points and values must have same length")
    if scale:
        pmin = points.min(axis=0)[None,:]
        vmin = values.min()
        pscale = np.abs(points).max(axis=0)[None,:] - pmin
        vscale = np.abs(values).max() - vmin
    else:
        pscale = np.ones((points.shape[1],), dtype=float)[None,:]
        vscale = 1.0
        pmin = np.zeros((points.shape[1],), dtype=float)[None,:]
        vmin = 0.0
    vand = vander((points - pmin) / pscale, deg)
    if scale_vand:
        sv = np.sqrt((vand*vand).sum(axis=0))
        vv = vand / sv[None,:]
    else:
        vv = vand
    coeffs = np.linalg.lstsq(vv, (values - vmin) / vscale, rcond=None)[0]
    if scale_vand:
        coeffs = coeffs / sv
    return {'coeffs': coeffs,
            'deg': deg, 'pscale': pscale, 'vscale': vscale,
            'pmin': pmin, 'vmin': vmin, 'ndim': points.shape[1]}


def polyval(fit, points, der=0):
    """Evaluate polynomial generated by :func:`polyfit` on `points`.

    Parameters
    ----------
    fit, points : see :func:`polyfit`
    der : int, optional
        Derivative order. Only for 1D, uses np.polyder().

    Notes
    -----
    For 1D we provide "analytic" derivatives using np.polyder(). For ND, we
    didn't implement an equivalent machinery. For 2D, you might get away with
    fitting a bispline (see Interpol2D) and use it's derivs. For ND, try rbf.py's RBF
    interpolator which has at least 1st derivatives for arbitrary dimensions.

    See Also
    --------
    :class:`PolyFit`, :class:`PolyFit1D`, :func:`polyfit`
    """
    assert points.ndim == 2, "points must be 2d array"
    assert (p_ndim := points.shape[1]) == (f_ndim := fit['ndim']), (
        f"points have wrong ndim: {p_ndim}, expect {f_ndim}")
    pscale, pmin = fit['pscale'], fit['pmin']
    vscale, vmin = fit['vscale'], fit['vmin']
    if der > 0:
        assert points.shape[1] == 1, "deriv only for 1d poly (ndim=1)"
        # ::-1 b/c numpy stores poly coeffs in reversed order
        dcoeffs = np.polyder(fit['coeffs'][::-1], m=der)
        return np.polyval(dcoeffs, (points[:,0] - pmin[0,0]) / pscale[0,0]) / \
            pscale[0,0]**der * vscale
    else:
        vand = vander((points - pmin) / pscale, fit['deg'])
        return np.dot(vand, fit['coeffs']) * vscale + vmin


class PolyFit:
    """High level interface to poly{fit,val}, similar to :class:`Spline`
    and :class:`Interpol2D`.

    Arguments and keywords to :meth:`__init__` are the same as for
    :func:`polyfit`. Keywords to :meth:`__call__` are same as for
    :func:`polyval`.

    Parameters
    ----------
    points : nd array (npoints,ndim)
        `npoints` points in `ndim`-space, to be fitted by a `ndim` polynomial
        f(x0,x1,...,x{ndim-1}).
    values : 1d array
    deg : int
        Degree of the poly (e.g. 3 for cubic).
    scale: bool, optional
        Scale `points` and `values` to unity internally before fitting.

    Notes
    -----
    | __init__: `points` must be (npoints,ndim) even if ndim=1.
    | __call__: `points` can be (ndim,) instead of (1,ndim), need this if called in
                fmin()

    Examples
    --------
    >>> fit1=polyfit(points, values, deg=3); polyval(fit1, new_points)
    >>> # the same
    >>> f1=PolyFit(points, values, 3); f1(new_points)
    """
    def __init__(self, points, values, *args, **kwds):
        """
        Parameters
        ----------
        points : nd array (npoints, ndim)
        values : 1d array (npoints,)
        **kwds : keywords to polyfit()
        """
        self.points = self._fix_shape_init(points)
        assert self.points.ndim == 2, "points is not 2d array"
        self.values = values
        self.fitfunc = polyfit
        self.evalfunc = polyval
        self.fit = self.fitfunc(self.points, self.values, *args, **kwds)

    @staticmethod
    def _fix_shape_init(points):
        return points

    @staticmethod
    def _fix_shape_call(points):
        assert hasattr(points, 'ndim'), "points must be an array with ndim attr"
        # (ndim,) -> (1,ndim) -> 1 point in ndim space
        if points.ndim == 1:
            return True, points[None,:]
        else:
            return False, points

    def __call__(self, points, **kwds):
        _got_single_point, points = self._fix_shape_call(points)
        ret = self.evalfunc(self.fit, points, **kwds)
        return ret[0] if _got_single_point else ret

    def get_min(self, x0=None, **kwds):
        """Minimize fit function by `scipy.optimize.fmin()`.

        Parameters
        ----------
        x0 : 1d array, optional
            Initial guess. If not given then `points[i,...]` at the min of
            `values` is used.
        **kwds : keywords to fmin()

        Returns
        -------
        1d array (ndim,)
        """
        _kwds = dict(disp=0, xtol=1e-12, ftol=1e-8, maxfun=1e4, maxiter=1e4)
        _kwds.update(kwds)
        if x0 is None:
            idx = self.values.argmin()
            x0 = self.points[idx,...]
        xopt = optimize.fmin(self, x0, **_kwds)
        return xopt


# Need to inherit first Fit1D such that Fit1D.get_min() is used instead of
# PolyFit.get_min().
class PolyFit1D(Fit1D, PolyFit):
    """1D special case version of :class:`PolyFit` which handles 1d and scalar
    `points` Also :meth:`get_min()` uses the root of the poly's 1st derivative
    instead of ``fmin()``.

    | __init__: points (npoints,1) or (npoints,)
    | __call__: points (npoints,1) or (npoints,) or scalar

    Examples
    --------
    >>> x=np.linspace(-5,5,10); y=(x-1)**2+1
    >>> f=num.PolyFit1D(x,y,2)
    >>> f(0)
    2.0000000000000009
    >>> f.get_min()
    1.0
    >>> xx = linspace(x[0],x[-1],50)
    >>> plot(x,y,'o', label='data')
    >>> plot(xx, f(xx), label='poly')
    >>> plot(xx, f(xx,der=1), label='d(poly)/dx')
    >>> legend()
    """
    def __init__(self, *args, **kwds):
        """
        Parameters
        ----------
        See PolyFit
        """
        # We need to exec PolyFit.__init__() here expliclity. Using super(...)
        # would call Fit1D.__init__().
        PolyFit.__init__(self, *args, **kwds)
        assert self.points.ndim == 2 and self.points.shape[1] == 1, \
            ("points has wrong shape: %s, expect (npoints,1)" \
            %str(self.points.shape))
        # set self.x, self.y, need that in Fit1D._findroot()
        Fit1D.__init__(self, self.points[:,0], self.values)


    @staticmethod
    def _fix_shape_init(points):
        pp = np.asarray(points)
        # 1 -> (1,1)
        if pp.ndim == 0:
            return np.array([[pp]])
        # (M,) -> (M,1)
        elif pp.ndim == 1:
            return pp[:,None]
        # (M,N)
        elif pp.ndim == 2:
            assert pp.shape[1] == self.fit['ndim'], "points have wrong ndim"
            return pp
        else:
            raise ValueError("points has wrong shape or dim")

    def _fix_shape_call(self, points):
        pp = np.asarray(points)
        # 1 -> (1,1)
        if pp.ndim == 0:
            return True, np.array([[pp]])
        else:
            if pp.ndim == 1:
                if len(pp) == self.fit['ndim']:
                    # (N,) -> (1,N)
                    return True, pp[None,:]
                else:
                    # (M,) -> (M,1)
                    return False, pp[:,None]
            elif pp.ndim == 2:
                assert pp.shape[1] == self.fit['ndim'], "points have wrong ndim"
                return False, pp
            else:
                raise ValueError("points has wrong shape or dim")


def match_mask(arr, values, fullout=False, eps=None):
    """Bool array of ``len(arr)`` which is True if ``arr[i] == values[j],
    j=0..len(values)``.

    This is the same as creating bool arrays like ``arr == some_value`` just
    that `some_value` can be an array (numpy can only do `some_value` =
    scalar). By default we assume integer arrays, unless `eps` is used.

    With `eps=None` and `fullout=False`, it behaves like ``numpy.in1d``.

    Parameters
    ----------
    arr : 1d array
    values : 1d array
        Values to be found in `arr`.
    fullout : bool
    eps : float
        Use this threshold to compare array values.

    Returns
    -------
    ret : fullout = False
    ret, idx_lst : fullout = True
    ret : 1d array, bool, len(arr)
        Bool mask array for indexing `arr`. ``arr[ret]`` pulls out all values
        which are also in `values`.
    idx_lst : 1d array
        Indices for which ``arr[idx_lst[i]]`` equals some value in `values`.

    Examples
    --------
    >>> arr=array([1,2,3,4,5]); values=array([1,3])
    >>> num.match_mask(arr, values)
    array([ True, False,  True, False, False], dtype=bool)
    >>> num.match_mask(arr, values, fullout=True)
    (array([ True, False,  True, False, False], dtype=bool), array([0, 2]))
    >>> arr[num.match_mask(arr, values)]
    array([1, 3])
    >>> # handle cases where len(values) > len(arr) and values not contained in arr
    >>> arr=array([1,2,3,4,5]); values=array([1,3,3,3,7,9,-3,-4,-5])
    >>> num.match_mask(arr, values, fullout=True)
    (array([ True, False,  True, False, False], dtype=bool), array([0, 2]))
    >>> # float values: use eps
    >>> num.match_mask(arr+0.1, values, fullout=True, eps=0.2)
    (array([ True, False,  True, False, False], dtype=bool), array([0, 2]))

    See Also
    --------
    numpy.in1d
    """
    assert arr.ndim == 1, ("arr must be 1d array")
    if eps is None:
        # assume integer array
        idx_lst = ((arr[None,:] - values[:,None]) == 0).nonzero()[1]
    else:
        idx_lst = (np.abs(arr[None,:] - values[:,None]) < eps).nonzero()[1]
    idx_lst = np.unique(idx_lst)
    ret = np.zeros((arr.shape[0],), dtype=bool)
    ret.put(idx_lst, True)
    if fullout:
        return ret, idx_lst
    else:
        return ret


def order_similar(arr, repeat=1, order=2):
    """Band ordering algorithm. Uses up to quadradic extrapolation. Handles
    crossing points.

    This can be used to order dispersion plots, for instance.

    Parameters
    ----------
    arr : 2d array (npoints, ndim)
        `ndim` 1d data streams with `npoints` each.
    repeat : int
        1: run 1 time, N: run recursively N times
    order : int
        Order of extrapolation: 0 = next similar point, 1 = linear using the
        two last points, 2 = quadratic using the 3 last points

    Returns
    -------
    arr2 : like `arr`
        Array with ordered data series.

    Notes
    -----
    The more points, the better. The first 1-2 steps should start smoothly or
    the algo will get confused. If you don't get all crossing points resolved,
    try ``repeat > 1``. But if the algo placed points from different data
    streams into one, you are lost. Then you can only use more points to make
    the extrapolation more precise.

    Examples
    --------
    >>> import numpy as np
    >>> from pwtools import mpl, num
    >>> plt = mpl.plt
    >>> x = np.linspace(0,10,200)
    >>> a = np.array([np.sin(0.5*x),
    ...               1.2*np.cos(2*x),
    ...               np.sin(2.5*(x-1.5)),
    ...               0.2*np.sin(x-1.1),
    ...               0.3*np.sin(x-1.1),
    ...               ]).T
    >>> for ai in a:
    ...     np.random.shuffle(ai)
    >>> plt.figure(); plt.plot(a); plt.title('raw data')
    >>> aa = num.order_similar(a, repeat=1, order=2)
    >>> plt.figure(); plt.plot(aa); plt.title('sorted, repeat=1, order=2')
    >>> aa = num.order_similar(a, repeat=2, order=1)
    >>> plt.figure(); plt.plot(aa); plt.title('sorted, repeat=2, order=1')
    >>> plt.show()
    """
    assert 0 <= order <= 2, "order must be 0,1,2"
    assert repeat >= 1, "repeat must be >= 1"
    _o_zero = False
    _o_one = False
    _o_two = False
    if order >= 0:
        _o_zero = True
    if order >= 1:
        _o_one = True
    if order == 2:
        _o_two = True
    if repeat == 1:
        ni,nj = arr.shape
        arr2 = np.empty_like(arr)
        arr2[0,:] = arr[0,:]
        for ii in range(ni-1):
            for jj in range(nj):
                if (ii == 0) and _o_zero:
                    # 1st row: choose next similar point
                    ref = arr2[ii,jj]
                elif (ii == 1) and _o_one:
                    # 2 rows: linear extrapolation
                    ref = 2*arr2[ii,jj] - arr2[ii-1,jj]
                elif (ii >= 2) and _o_two:
                    # > 2 rows: quadradic extrapolation
                    #
                    # Calling polyfit is pretty slow and could be
                    # optimized. For 3 points in x,y, we can write the
                    # expressions for the coeffs by hand, yes?
                    x = np.array([0,1,2])
                    y = arr2[ii-2:ii+1,jj]
                    p = np.polyfit(x,y,2)
                    ref = np.polyval(p,3)
                dif = np.abs(arr[ii+1,:] - ref)
                idx = np.argmin(dif)
                arr2[ii+1,jj] = arr[ii+1,idx]
    else:
        arr2 = order_similar(arr, repeat=repeat-1)
    return arr2


def round_up_next_multiple(x, mult):
    """Round integer `x` up to the next possible multiple of `mult`."""
    rem = x % mult
    if rem > 0:
        return x - rem  + mult
    else:
        return x

def norm(a):
    """2-norm for real vectors."""
    assert len(a.shape) == 1, "input must be 1d array"
    # math.sqrt is faster than np.sqrt for scalar args
    return sqrt(np.dot(a,a))


def meshgridt(x, y):
    """Shortcut for ``numpy.meshgrid(x, y, indexing="ij")``

    A version of ``X,Y = numpy.meshgrid(x,y)`` which returns X and Y
    transposed, i.e. (nx, ny) instead (ny, nx) where nx,ny = len(x),len(y).

    This is useful for dealing with 2D splines in
    scipy.interpolate.bisplev(), which also returns a (nx,ny) array.

    Parameters
    ----------
    x,y : 1d arrays
    """
    ##X,Y = np.meshgrid(x,y)
    ##return X.T, Y.T
    return np.meshgrid(x, y, indexing="ij")


def euler_matrix(phi, theta, psi, deg=False):
    r"""Euler's rotation matrix.

    We use the x-convention, as in [1]_.

    .. math::
        (\phi, \theta, \psi)        \\
        A = B\,C\,D                 \\
        D: \phi = 0,...,2\,\pi      \\
        C: \theta = 0,...,\pi       \\
        B: \psi = 0,...,2\,\pi      \\

    Parameters
    ----------
    phi, theta, psi : float
        angles
    deg : bool
        angles in degree (True) or radians (False, default)

    References
    ----------
    .. [1] http://mathworld.wolfram.com/EulerAngles.html
    """
    if deg:
        phi = radians(phi)
        theta = radians(theta)
        psi = radians(psi)
    assert abs(phi) <= 2*pi
    assert abs(theta) <= pi
    assert abs(psi) <= 2*pi
    sin_a = sin(phi)
    sin_b = sin(theta)
    sin_c = sin(psi)
    cos_a = cos(phi)
    cos_b = cos(theta)
    cos_c = cos(psi)
    D = np.array([[ cos_a,  sin_a,      0],
                  [-sin_a,  cos_a,      0],
                  [     0,      0,      1]])*1.0

    C = np.array([[     1,      0,      0],
                  [     0,  cos_b,  sin_b],
                  [     0, -sin_b,  cos_b]])*1.0

    B = np.array([[ cos_c,  sin_c,      0],
                  [-sin_c,  cos_c,      0],
                  [     0,      0,      1]])*1.0
    return np.dot(B, np.dot(C, D))

