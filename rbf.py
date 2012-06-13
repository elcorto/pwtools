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
# 0...1e4), you may get bad interpolation between points. This is b/c the RPFs
# also live on more or less equal x-y scales.

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.linalg as linalg
from scipy.spatial import distance
from pwtools import flib_wrap
from pwtools import mpl, debug
rand = np.random.rand

# Used for timing.
##dbg = debug.Debug()

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
        Y: 1d array, (M,) or 2d (M,1)
            function values at training points
        C : 2d array (K,N), optional
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
        Simple 1D example.
        >>> X=linspace(0,10,20)[:,None]
        >>> Y=sin(X)
        >>> rbfi=rbf.RBFInt(X,Y)
        >>> rbfi.train()
        >>> XI=linspace(0,10,100)
        >>> plot(X,Y,'o', label='data')
        >>> plot(XI, sin(XI), label='sin(x)')
        >>> plot(XI, rbfi(XI[:,None]), label='rbf')
        >>> legend()
        """
        self.X = X
        self.Y = Y
        self.C = X if C is None else C
        self.rbf = rbf
        self.verbose = verbose
        self.distmethod = distmethod

        self.Rsq = None
    
    def msg(self, msg):
        """Print a message."""
        if self.verbose:
            print(msg)

    def calc_dist_mat(self):
        """Calculate self.Rsq for the training set. 
        
        Can be used in conjunction w/ set_Y() to construct a network which
        calculates Rsq only once. Then it can be trained for different Y many
        times, using the same X grid.
        
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
        self.Y = Y

    def get_dist_mat(self, X=None, C=None):
        """Matrix of distance values r_ij = ||x_i - c_j||.
        x_i : X[i,:]
        c_i : C[i,:]

        args:
        -----
        X : None or array (M,N) with n-dim points
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
            self.msg("----get_dist_mat...")
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

    def get_rbf_mat(self, Rsq):
        """Matrix of RBF values g_ij = rbf(||x_i - c_j||)."""
        self.msg("----get_rbf_mat...")
        G = self.rbf(Rsq)
        return G
    
    def get_param(self, param):
        """Return `param` for RPF.

        args:
        -----
        param : string 'est' or float
            If 'est', then return the mean distance of all points. 
        """
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
        self.msg("--train_linalg...")
        # this test may be expensive for big data sets
        assert (self.C == self.X).all(), "C == X not fulfilled"
        if self.Rsq is None:
            self.Rsq = self.get_dist_mat(X=self.X, C=self.C)
        self.rbf.param = self.get_param(param)
        G = self.get_rbf_mat(Rsq=self.Rsq)
        weights = linalg.solve(G, self.Y)
        return {'weights': weights, 'param': None}
    
    def _train_root(self, param='est'):
        """Optimize weights by calculating the root of this error function:
            
            >>> rbfi = RBFInt(...)
            >>> error(weights) = Y - rbfi.interpolate(X)
        
        This is only proof of concept. Use train_linalg.

        We tried fmin() and fmin_bfgs() both w/ start weights from
        train_linalg() or simply ones(). They converge extremly slowly. Why?
        Maybe we need to provide analytical gradients. This seems to be a very
        flat minimum. Even starting from almost-perfect train_linalg() weights
        doesn't help. Maybe the problem is too high-dimensional (M).

        But using a n-dim root finder instead works (scipy.optimize.anderson()
        or broyden1() or broyden2()). Starting from train_linalg() weights, we
        converge in 3 steps back to the same weights. If we use ones(),
        convergence is slow, again. => train_linalg() is optimal and doesn't
        need to be improved.

        can update:
        -----------
        weights

        args:
        -----
        param : param

        returns:
        --------
        {'weights': w, 'param': p}
            w : (K,), linear output weights for K center vectors
            p : None
        """
        def func_fmin(weights):
            self.weights = weights
            d = self.Y - self.interpolate()
            err = math.sqrt(np.dot(d,d))
            self.msg('train_fmin: err=%e' %err)
            return err
        def func_root(weights):
            self.weights = weights
            d = self.Y - self.interpolate()
            err = math.sqrt(np.dot(d,d))
            self.msg('train_fmin: err=%e' %err)
            return d
        # Set self.Rsq once. Then it is used in each interpolate() call.
        self.Rsq = self.get_dist_mat(self.X)
        weights0 = self.train_linalg(param=param)['weights']
##        weights0 = np.ones((len(self.Y),))
##        self.rbf.param = self.get_param(param)
##        weights = opt.fmin(func_fmin, weights0)
        weights = opt.anderson(func_root, weights0)
        return {'weights': self.weights, 'param': None}
    
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
    
    def interpolate(self, X=None):
        """Actually do interpolation. Return interpolated Y values at each
        point in X.
        
        args:
        -----
        X : 2d array (L,N) or 1d array (N,) or None 
            L data points in N-dim space. For each point, return interpolated
            value. If None, then X = self.X, i.e. just evaluate the training
            set. If 1d, then this is just 1 point, which will be converted to
            shape (1,N) internally.

        returns:
        --------
        YI : 1d array (L,)

        notes:
        ------
        Calculates 
            Rsq = (X - C)**2      # squared distance matrix
            G = rbf(Rsq)          # RBF values
            YI = dot(G, weights)  # interpolation y_i = Sum_j g_ij w_j
        """
        if X is not None:
            X = np.asarray(X)
            if len(X.shape) == 1:
                X = X[None,:]
        Rsq = self.get_dist_mat(X=X, C=self.C)
        G = self.get_rbf_mat(Rsq)
        assert G.shape[1] == len(self.weights), \
               "shape mismatch between g_ij: %s and w_j: %s, first dim of "\
               "g_ij must match length of w_j" %(str(G.shape), \
               str(self.weights.shape))
        # normalize weights
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        ret = np.dot(G, ww / maxw) * maxw
        return ret

    def __call__(self, *args, **kwargs):
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

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class SurfaceData(object):
    def __init__(self, xlim, ylim, nx, ny, mode):
        self.xlim = xlim
        self.ylim = ylim
        self.nx = nx
        self.ny = ny
        self.xg, self.yg = self.get_xy_grid()
        self.XG, self.YG = mpl.meshgridt(self.xg, self.yg)
        self.X = self.gen_coords(mode)
        
    def gen_coords(self, mode='grid'):
        if mode =='grid':
            X = np.empty((self.nx * self.ny,2))
            X[:,0] = self.XG.flatten()
            X[:,1] = self.YG.flatten()
            ##for i in range(self.nx):
            ##    for j in range(self.ny):
            ##        X[i*self.ny+j,0] = self.xg[i]
            ##        X[i*self.ny+j,1] = self.yg[j]
        elif mode == 'rand':
            X = rand(self.nx * self.ny, 2)
            X[:,0] = X[:,0] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            X[:,1] = X[:,1] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return X
    
    def get_xy_grid(self):
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        return x,y

    def get_X(self, X=None):
        return self.X if X is None else X

    def func(self, X=None):
        X = self.get_X(X)
        return None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MexicanHat(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        r = np.sqrt((X**2).sum(axis=1))
        return np.sin(r)/r

class UpDown(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        x = X[:,0]
        y = X[:,1]
        return x*np.exp(-x**2-y**2)

class SinExp(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        x = X[:,0]
        y = X[:,1]
        return np.sin(np.exp(x)) * np.cos(y) + 0.5*y

class Square(SurfaceData):
    def func(self, X=None):
        X = self.get_X(X)
        x = X[:,0]
        y = X[:,1]
        return (x**2 + y**2)

if __name__ == '__main__':
    
    #--- 2d case ----------------------
    
    fu = MexicanHat([-10,20], [-10,15], 15, 15, 'rand')
##    fu = UpDown([-2,2], [-2,2], 20, 20, 'grid')
##    fu = SinExp([-1,2.5], [-2,2], 40, 30, 'rand')
##    fu = Square([-1,1], [-1,1], 20, 20, 'grid')
    
    X = fu.X
    Z = fu(X)

    rbfi = RBFInt(X, Z, X, rbf=RBFMultiquadric(), verbose=True)
    print "train linalg single ..."
    rbfi.train('linalg')
    print "param:", rbfi.rbf.param
    print "... ready train"
    
    print "train param ..."
    rbfi = train_param(X, Z, pattern='rand', randskip=0.2)
    print "param:", rbfi.rbf.param
    print "... ready train"
    
    dati = SurfaceData(fu.xlim, fu.ylim, fu.nx*2, fu.ny*2, 'grid')

    ZI_func = fu(dati.X)
    ZI_rbf = rbfi(dati.X)
    ZG_func = ZI_func.reshape((dati.nx, dati.ny))
    ZG_rbf = ZI_rbf.reshape((dati.nx, dati.ny))
    zlim = [ZI_func.min(), ZI_func.max()]

    fig, ax = mpl.fig_ax3d()
    ax.scatter(X[:,0],X[:,1],Z, color='r')
    dif = np.abs(ZI_func - ZI_rbf).reshape((dati.nx, dati.ny))
    wf = ax.plot_wireframe(dati.XG, dati.YG, ZG_rbf, cstride=1, rstride=1, color='g')
    wf.set_alpha(0.5)
    wf2 = ax.plot_wireframe(dati.XG, dati.YG, ZG_func, cstride=1, rstride=1, color='m')
    wf2.set_alpha(0.5)
    cont = ax.contour(dati.XG, dati.YG, dif, offset=zlim[0], 
                      levels=np.linspace(dif.min(), dif.max(), 20))
    fig.colorbar(cont, aspect=5, shrink=0.5, format="%.3g")    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim3d(dati.xlim)
    ax.set_ylim3d(dati.ylim)
    ax.set_zlim3d(zlim)

    plt.show()
