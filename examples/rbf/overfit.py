#!/usr/bin/python3

"""
Pathological case of overfitting (interpolation of all points) when p is chosen
too small. 

We use Rbf in least squares mode where r is None, but the same results can be
obtained with a very small r, such as r=1e-11 (weak smoothing).

There is a global minimum of the fit error at very small p (narrow RBFs can fit
all points). It turns out that this is almost exactly the scipy estimate of p
in scipy.interpolate.Rbf, which is good for interpolation, but not for fitting.

Note that the fit error landscape is pretty rugged and flat around the default
p (mean distance estimate) such that a local optimizer such as 'fmin' won't
produce anything more meaningful. We use method='de' (differential_evolution)
to make sure we find the global minimum for demonstration purpose.

This example shows that it is paramount to use cross-validation to calculate a
more useful fit error estimate. Observe how the CV error is highest where the
normal fit error has the global minimum (location of scipy's p estimate).

Note that we use a low number of points here for speed, such that the CV error
is not yet converged, but shows the correct trend already.
"""

import numpy as np
from pwtools import mpl, rbf
plt = mpl.plt


if __name__ == '__main__':
##    rbf_kwds = dict(rbf='inv_multi', r=1e-11)
    rbf_kwds = dict(rbf='inv_multi', r=None)

    fig, axs = plt.subplots(2, 1)
    x = np.linspace(0, 10, 100)
    rnd = np.random.RandomState(seed=123)
    y = np.sin(x) + rnd.rand(len(x))
    points = x[:,None]
    values = y
    xi = np.linspace(x[0], x[-1], len(x)*4)[:,None]
    
    fe = rbf.FitError(points, values, rbf_kwds=rbf_kwds)

    ax = axs[0]
    ax.plot(x, y, 'o', alpha=0.3)

    rbfi = rbf.Rbf(points, values, **rbf_kwds)
    p_default = rbfi.get_params()[0]
    ax.plot(xi, rbfi(xi), 'r', label='$p$ default')
    
    rbfi = rbf.fit_opt(points, values, method='de', what='p',
                       opt_kwds=dict(bounds=[(0.1, 10)], maxiter=10), 
                       rbf_kwds=rbf_kwds,
                       cv_kwds=None)
    p_opt = rbfi.get_params()[0]
    ax.plot(xi, rbfi(xi), 'g', label='$p$ opt')
    
    rbfi = rbf.Rbf(points, values, p=10, **rbf_kwds)
    p_cv = rbfi.get_params()[0]
    ax.plot(xi, rbfi(xi), 'y', label='$p$ big')
    
    rbfi = rbf.Rbf(points, values, p=rbf.estimate_p(points, 'scipy'), 
                      **rbf_kwds)
    p_scipy = rbfi.get_params()[0]
    ax.plot(xi, rbfi(xi), 'm', label='$p$ scipy')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    ax = axs[1]
    ax2 = ax.twinx()
    params = np.linspace(1e-8, 15, 50)
    ax.plot(params, [fe.err_direct([px]) for px in params], 
            label='fit error')
    ax2.plot(params, [fe.err_cv([px]) for px in params], 'k', 
             label='CV fit error')
    ax.plot([p_default], fe.err_direct([p_default]), 'ro', 
            label='$p$ default')
    ax.plot([p_scipy], fe.err_direct([p_scipy]), 'mo', 
            label='$p$ scipy')
    ax.plot([p_opt], fe.err_direct([p_opt]), 'go', 
            label='$p$ opt')
    ax.plot([p_cv], fe.err_direct([p_cv]), 'yo', 
            label='$p$ big')
    ax2.set_ylim(0, 1)
    ax.set_xlabel('$p$')
    ax.set_ylabel('fit error')
    ax2.set_ylabel('CV fit error')
    ax.legend(*mpl.collect_legends(ax,ax2), loc='lower right')

    plt.subplots_adjust(hspace=0.33)
    plt.show()
