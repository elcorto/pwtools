# Radial Basis Functions Neural Network for interpolation of n-dim data points.
#
# Refs:
# [1] http://en.wikipedia.org/wiki/Artificial_neural_network#Radial_basis_function_.28RBF.29_network
# [2] http://en.wikipedia.org/wiki/Radial_basis_function_network
# [3] NR, 3rd ed., ch 3.7
#
# Training
# --------
# One guy from the Behler group said they train their NN w/ a Kalman filter.
#
# For our RBF network, we use the traditional approach and simply solve a
# linear system for the weights. Calculating the distance matrix w/
# scipy.spatial.distance.cdist() or flib_wrap.distsq() is handled efficiently,
# even for many points. But we get into trouble for many points (order 1e4) b/c
# solving a big dense linear system (1e4 x 1e4) with plain numpy.linalg on a
# single core is possible but painful (takes some minutes) -- the traditional
# RBF problem. Maybe use numpy build against threaded MKL, or even scalapack?
# For the latter, look at how GPAW does this.
#
# There are some other projects out there:
# 
# * scipy.interpolate.Rbf 
#   Essentially the same as we do. We took some ideas from there.
# * http://pypi.python.org/pypi/PyRadbas/0.1.0
#   Seems to work with gaussians hardcoded. Not what we want.
# * http://code.google.com/p/pyrbf/
#   Seems pretty useful for large problems (many points), but not 
#   documented very much.
# 
# rbf parameter
# -------------
# Each RBF has a single parameter (`param`), which can be tuned. This is
# usually a measure for the "width" of the function.
#
# * What seems to work best is RBFMultiquadric + train('linalg') +
#   param='est' (mean distance of all points), i.e. the "traditional" RBF
#   approach. This is exactly what's done in scipy.interpolate.Rbf .
#
# * It seems that for very few data points (say 5x5 for x**2 + x**2), the
#   default param='est' is too small, which results in wildly fluctuating
#   interpolation, b/c there is no "support" in between the data points. We
#   need to have wider RBFs. Usually, bigger (x10), sometimes much bigger
#   (x100) params work.
# 
# * However, we also had a case where we found by accident that the square of a
#   small param (param (from 'est') = 0.3, param**2 = 0.1) was actually much
#   better than the one from param='est'.
#
# other traing algos
# -------------------
# * train_param(), i.e. optimizing the RBF's free parameter by repeated
#   training, works extremely good, for all kinds of (random) training
#   point pattern on all test problems, but not so much for real world 
#   data, when points are slightly noisy.
# * If you really use train_param(), then make sure to use train_param(...,
#   ftol=1e-8, maxfun=...), i.e. pass fmin() keywords. The default fmin()
#   convergence thresholds are pretty high.
# * We really need a training which lets us specify some kind of smoothness.
#   For perfect data, RBF works perfect. But for "almost" perfect data, we
#   would like to do some kind of fit instead, like the "s" parameter to
#   scipy.interpolate.bisplrep().
#
# input data
# ----------
# It usually helps if all data ranges (points X and values Y) are in the same
# order of magnitude, e.g. all have a maximum close to unity or so. If, for
# example, X and Y have very different scales (say X -0.1 .. 0.1 and Y
# 0...1e4), you may get bad interpolation between points. This is b/c the RBFs
# also live on more or less equal x-y scales.

import math
import numpy as np
import scipy.optimize as opt
import scipy.linalg as linalg
from scipy.spatial import distance
from pwtools import flib_wrap
from pwtools import timer

##TT = timer.TagTimer()

class RBFFunction(object):
    """Represent a single radial basis function."""
    def __init__(self, param=None):
        """
        args:
        -----
        param : scalar or None
            The RBF's free parameter.
        """
        self.param = param
    
    def _get_param(self, param):
        p = self.param if param is None else param
        assert p is not None, ("param is None")
        return p

    def __call__(self, rsq, param=None):
        """
        args:
        -----
        rsq : scalar or nd array
            Squared distances. Squared b/c this avoids using sqrt() in the
            distance matrix (R) calculation just to square them here again.
            bad:  R   = sqrt((X - C)**2)
            good: Rsq = (X - C)**2
            Note that with scipy.spatial, we get R instead of Rsq anyway.
        param : scalar or None, optional
            If None, then self.param is used.
        """
        pass

class RBFGauss(RBFFunction):
    def __call__(self, rsq, param=None):
        return np.exp(-0.5*rsq / self._get_param(param)**2.0)

class RBFMultiquadric(RBFFunction):
    def __call__(self, rsq, param=None):
        return np.sqrt(rsq + self._get_param(param)**2.0)

class RBFInverseMultiquadric(RBFFunction):
    def __call__(self, rsq, param=None):
        return (rsq + self._get_param(param)**2.0)**(-0.5)

class RBFInt(object):
    def __init__(self, X, Y=None, C=None, rbf=RBFMultiquadric(),
                 verbose=False, distmethod='fortran'):
        """
        args:
        -----
        X : 2d array, (M,N)
            data points : M points in N-dim space, training set points
        Y: 1d array, (M,)
            function values at training points
        C : 2d array (K,N)
            K N-dim center vectors, for usual interpolation training X == C
            (default)
        rbf : RBFFunction instance, optional
        verbose : bool, optional
            print some messages
        distmethod : str, optional
            Choose method for distance matrix calculation.
                fortran : Fortran implementation (OpenMP, fastest)
                spatial : scipy.spatial.distance.cdist
                numpy : pure numpy (memory hungry, slowest)

        example:
        --------
        1D example w/ derivatives. For 1d, we need to use X[:,None] b/c the
        input array containing training (X) and interpolation (XI) points must
        be 2d.
        >>> X=linspace(0,10,20)     # shape (M,), M=20 points
        >>> Y=sin(X)                # shape (M,)
        >>> rbfi=rbf.RBFInt(X[:,None],Y)    
        >>> rbfi.train()
        >>> XI=linspace(0,10,100)   # shape (M,), M=100 points
        >>> plot(X,Y,'o', label='data')
        >>> plot(XI, sin(XI), label='sin(x)')
        >>> plot(XI, rbfi(XI[:,None]), label='rbf')
        >>> plot(XI, cos(XI), label='cos(x)')
        >>> plot(XI, rbfi(XI[:,None],der=1)[:,0], label='d(rbf)/dx')
        >>> legend()
        2D example
        >>> from pwtools import mpl, rbf
        >>> x1=linspace(-3,3,10); x2=x1
        >>> X1,X2=np.meshgrid(x1,x2); X1,X2 = X1.T,X2.T
        >>> X = np.array([X1.flatten(), X2.flatten()]).T
        >>> Y = (np.sin(X1)+np.cos(X2)).flatten()
        >>> x1i=linspace(-3,3,50); x2i=x1i
        >>> X1I,X2I=np.meshgrid(x1i,x2i); X1I,X2I = X1I.T,X2I.T
        >>> XI = np.array([X1I.flatten(), X2I.flatten()]).T
        >>> rbfi=rbf.RBFInt(X,Y)
        >>> rbfi.train()
        >>> fig1,ax1 = mpl.fig_ax3d()
        >>> ax1.scatter(X[:,0], X[:,1], Y)
        >>> ax1.plot_wireframe(X1I, X2I, rbfi(XI).reshape(50,50))
        >>> fig2,ax2 = mpl.fig_ax3d()
        >>> ax2.plot_wireframe(X1I, X2I, rbfi(XI,der=1)[:,0].reshape(50,50),
        ...                    color='b', label='d/dx')
        >>> ax2.plot_wireframe(X1I, X2I, rbfi(XI,der=1)[:,1].reshape(50,50),
        ...                    color='r', label='d/dy')
        >>> ax2.legend()
        """
        self.X = X
        self.Y = Y
        self.C = self.X if C is None else C
        self.rbf = rbf
        self.verbose = verbose
        self.distmethod = distmethod
        self._assert_ndim_X(self.X)
        self._assert_ndim_X(self.C)
        self._assert_ndim_Y(self.Y)

        self.Rsq = None
    
    @staticmethod
    def _assert_ndim_X(X):
        assert X.ndim == 2, ("X or C not 2d array")

    @staticmethod
    def _assert_ndim_Y(Y):
        assert Y.ndim == 1, ("Y not 1d array")
    
    def msg(self, msg):
        """Print a message."""
        if self.verbose:
            print(msg)

    def calc_dist_mat(self):
        """Calculate self.Rsq for the training set. 
        
        Can be used in conjunction w/ set_Y() to construct a network which
        calculates Rsq only once. Then it can be trained for different Y many
        times, using the same X grid w/o re-calculating Rsq over and over
        again.
        
        example:
        --------
        >>> rbfi = rbf.RBFInt(X=X, Y=None)
        >>> rbfi.calc_dist_mat()
        >>> for Y in ...:
        >>> ... rbfi.set_Y(Y)
        >>> ... rbfi.train()
        >>> ... YI=rbfi(XI)
        """
        self.Rsq = self.get_dist_mat(X=self.X, C=self.C)
    
    def set_Y(self, Y):
        self._assert_ndim_Y(Y)
        self.Y = Y

    def get_dist_mat(self, X=None, C=None):
        """Matrix of distance values r_ij = ||x_i - c_j||.
        x_i : X[i,:]
        c_i : C[i,:]

        args:
        -----
        X : None or array (M,N) with N-dim points
            Training data or interpolation points. If None then self.Rsq is
            returned.
        C : 2d array (K,N), optional
            If None then self.C is used.

        returns:
        --------
        Rsq : (M,K), where K = M usually for training
        """
        # pure numpy:
        #     dist = X[:,None,...] - C[None,...]
        #     Rsq = (dist**2.0).sum(axis=-1)
        # where   
        #     X:    (M,N)
        #     C:    (K,N)
        #     dist: (M,K,N) "matrix" of distance vectors (only for numpy case)
        #     Rsq:  (M,K)    matrix of squared distance values
        # Creates *big* temorary arrays if X is big (~1e4 points).
        #
        # training:
        #     If X == C, we could also use pdist(X), which would give us a 1d
        #     array of all distances. But we need the redundant square matrix
        #     form for get_rbf_mat() anyway, so there is no real point in
        #     special-casing that. These two are the same:
        #      >>> R = spatial.squareform(spatial.distances.pdist(X))
        #      >>> R = spatial.distances.cdist(X,X)
        #      >>> Rsq = R**2
        if X is not None:
            self.msg("get_dist_mat...")
            C = self.C if C is None else C
            if self.distmethod == 'spatial':
                Rsq = distance.cdist(X, C)**2.0
            elif self.distmethod == 'fortran':                
                Rsq = flib_wrap.distsq(X,C)
            elif self.distmethod == 'numpy':                
                dist = X[:,None,...] - C[None,...]
                Rsq = (dist**2.0).sum(axis=-1)
            else:
                raise StandardError("unknown value for distmethod: %s"
                    %self.distmethod)
            return Rsq
        else:
            assert self.Rsq is not None, ("self.Rsq is None")
            return self.Rsq

    def get_rbf_mat(self, Rsq=None):
        """Matrix of RBF values g_ij = rbf(||x_i - c_j||)."""
        self.msg("get_rbf_mat...")
        Rsq = self.Rsq if Rsq is None else Rsq
        G = self.rbf(Rsq)
        return G
    
    def get_param(self, param):
        """Return `param` for RBF (to set self.rbf.param).

        args:
        -----
        param : string 'est' or float
            If 'est', then return the mean distance of all points. 
        """
        self.msg("get_param...")
        if param == 'est':
            Rsq = self.get_dist_mat()
            return np.sqrt(Rsq).mean()
        else:
            return param

    def _train_linalg(self, param='est'):
        """Simple training by solving for the weights w_j:
            y_i = Sum_j g_ij w_j
        in case C == X (center vectors are all data points). Then G is
        quadratic w/ dim (M,M).

        By definition, this always yields perfect interpolation at the points
        in X, but may oscillate between points if you have only a few. Use
        bigger param in that case.

        This method is a sure-fire one, i.e. it always works by definition. You
        "only" need to choose `param` properly (which can be very hard
        sometimes).
        
        can update:
        -----------
        weights

        returns:
        --------
        {'weights': w, 'param': p}
            w : (K,), linear output weights for K center vectors
            p : None
        """
        self.msg("_train_linalg...")
        # this test may be expensive for big data sets
        assert (self.C == self.X).all(), "C == X not fulfilled"
        if self.Rsq is None:
            self.Rsq = self.get_dist_mat(X=self.X, C=self.C)
        self.rbf.param = self.get_param(param)
        G = self.get_rbf_mat(Rsq=self.Rsq)
        weights = linalg.solve(G, self.Y)
        return {'weights': weights, 'param': None}
    
    def train(self, mode='linalg', *args, **kwargs):
        """Call one of the training methods self._train_<mode>().
        
        Always use this, not the methods directly, as they don't set
        self.weights after training!
        """
        func = getattr(self, '_train_' + mode)
        ret = func(*args, **kwargs)
        self.weights = ret['weights']
        if ret['param'] is not None:
            self.rbf.param = ret['param']
    
    def interpolate(self, X):
        """Actually do interpolation. Return interpolated Y values at each
        point in X.
        
        args:
        -----
        X : see interpolate()

        returns:
        --------
        YI : 1d array (L,)

        notes:
        ------
        Calculates 
            Rsq = (X - C)**2      # squared distance matrix
            G = rbf(Rsq)          # RBF values
            YI = dot(G, w)        # interpolation y_i = Sum_j g_ij w_j
        """
        self._assert_ndim_X(X)
        Rsq = self.get_dist_mat(X=X, C=self.C)
        G = self.get_rbf_mat(Rsq)
        assert G.shape[1] == len(self.weights), \
               "shape mismatch between g_ij: %s and w_j: %s, 2nd dim of "\
               "g_ij must match length of w_j" %(str(G.shape), \
               str(self.weights.shape))
        # normalize weights
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        ret = np.dot(G, ww / maxw) * maxw
        return ret
    
    def deriv(self, X):
        """Matrix of partial first derivatives.

        Implemented for 
            RBFMultiquadric
            RBFGauss

        args:
        -----
        X : see interpolate()

        returns:
        --------
        2d array (L,N)
            Each row holds the partial derivatives of one input point ``X[i,:] =
            [x_0, ..., x_N]``. For all points X: (L,N), we get the matrix::

            [[dy/dx0_0, dy/dx0_1, ..., dy/dx0_N],
             [...],
             [dy/dxL_0, dy/dxL_1, ..., dy/dxL_N]]
        """
        # For the implemented RBF types, the derivatives w.r.t. to the point
        # coords simplify to nice dot products, which can be evaluated
        # reasonably fast w/ numpy. We don't need to change the RBF's
        # implementations to provide a deriv() method. For that, they would
        # need to take X and C explicitely as args instead of squared
        # distances, which are calculated fast by Fortran outside.
        #
        # Speed:
        # We have one python loop over the L points (X.shape=(L,N)) left, so
        # this gets slow for many points.
        # 
        # Loop versions (for RBFMultiquadric):
        #
        # # 3 loops:                
        # D = np.zeros((L,N), dtype=float)
        # for zz in range(L):
        #     for kk in range(N):
        #         for ii in range(len(self.weights)):
        #             D[zz,kk] += (X[zz,kk] - C[ii,kk]) / G[zz,ii] * \
        #                 self.weights[ii]
        # 
        # # 2 loops:
        # D = np.zeros((L,N), dtype=float)
        # for zz in range(L):
        #     for kk in range(N):
        #         vec = -1.0 * (C[:,kk] - X[zz,kk]) / G[zz,:]
        #         D[zz,kk] = np.dot(vec, self.weights)
        self._assert_ndim_X(X)
        L,N = X.shape
        C = self.C
        Rsq = self.get_dist_mat(X=X, C=C)
        G = self.get_rbf_mat(Rsq)
        D = np.empty((L,N), dtype=float)
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        if isinstance(self.rbf, RBFMultiquadric):
            for zz in range(L):
                D[zz,:] = -np.dot(((C - X[zz,:]) / G[zz,:][:,None]).T, ww / maxw) * maxw
        elif isinstance(self.rbf, RBFGauss):
            for zz in range(L):
                D[zz,:] = 1.0 / self.rbf.param**2.0 * \
                    np.dot(((C - X[zz,:]) * G[zz,:][:,None]).T, ww / maxw) * maxw
        else:
            raise StandardError("derivative for rbf type not implemented")
        return D                

    def __call__(self, *args, **kwargs):
        """
        Call self.interpolate() or self.deriv().

        args:
        -----
        X : 2d array (L,N) or 1d array (N,) or None 
            L data points in N-dim space. If None, then X = self.X, i.e. just
            evaluate the training set. If 1d, then this is just 1 point, which
            will be converted to shape (1,N) internally.
        der : int
            If == 1 return matrix of partial derivatives (see self.deriv()), else
            interpolated Y values (default).

        returns:
        --------
        YI : 1d array (L,)
            Interpolated values.
        or             
        D : 2d array (L,N)
            1st partial derivatives.
        """
        if 'der' in kwargs.keys():
            if kwargs['der'] != 1:
                raise StandardError("only der=1 supported")
            kwargs.pop('der')
            return self.deriv(*args, **kwargs)
        else:            
            return self.interpolate(*args, **kwargs)


def train_param(X, Y, param0='est', pattern='rand', regstep=2,
                randskip=0.2, shuffle=False, rbf=RBFMultiquadric(), **fmin_kwds):
    """
    This could be also implemented as another training method inside RBFInt.
    Here, define a function which optimizes `param` by fmin using only a subset
    of the training set, evaluating the full training set and measuring the
    interpolation error at the missing points.

    args:
    -----
    X, Y: See RBFInt.
    param0 : float or str 'est'
        Start RBF parameter. 'est' for using the mean point distance in the
        training point set.
    pattern : str
        Pattern for picking out training points from X.
        'reg': Use a "regular" pattern, i.e. use only every `regstep`th point.
             Note that of course this depends on how points in X are sorted.
        'rand': Pick points at random, leaving out a fraction of `randskip`
            points. Example: randskip=0.2 means use ramdomly 80% of all points. 
    regstep : int
    randskip : float in [0,1]
    shuffle : randomize sort order of points in X *before* using `pattern`.
        Note that is for testing and in fact somewhat redundant, for instance
            shuffle=True  + pattern='regular' + regstep=2
        is the same as
            shuffle=False + pattern='random' + randskip=0.5
    rbf : RBFFunction instance
    **fmin_kwds : passed to fmin()

    returns:
    --------
    RBFInt instance
    """
    npoints = X.shape[0]
    assert 0.0 < randskip < 1.0, ("use 0.0 < randskip < 1.0")                
    assert 0 < regstep < npoints, ("use 0 < regstep < npoints")               
    def func(p, *args):
        rbfi.train('linalg', param=p[0])
        d = Y - rbfi(X)
        err = math.sqrt(np.dot(d,d))
        print('train_param: err=%e' %err)
        return err
    idxs = range(npoints)
    if shuffle:        
        np.random.shuffle(idxs)
    if pattern == 'reg':
        Xtr = X[idxs[::regstep],...]
        Ytr = Y[idxs[::regstep]]
    elif pattern == 'rand':
        msk = (np.random.rand(npoints) >= randskip).astype(bool)
        Xtr = X[np.array(idxs)[msk],...]
        Ytr = Y[np.array(idxs)[msk]]
    else:
        raise StandardError("unknown pattern")
    rbfi = RBFInt(Xtr, Ytr, rbf=rbf)
    rbfi.calc_dist_mat() # only for get_param .. fix later
    p0 = rbfi.get_param(param0) 
    popt = opt.fmin(func, [p0], args=(X,Y), **fmin_kwds)
    # After optimization, re-train w/ all points, using the optimized param.
    # Not sure if we should set param < popt b/c the full points set has
    # less mean distance. For test cases, this doesn't seem to make much
    # difference. Even not for randskip=0.5 (use only half the points) where we
    # would expect that the true `param` would be param/2.
    rbfi.train('linalg', param=popt[0])
    return rbfi

