#!/usr/bin/env python3

"""
Compare hyper parameter optimization of pwtools KRR and sklearn GP
==================================================================

pwtools.rbf.Rbf(..., p=..., r=...)
----------------------------------
Does Kernel Ridge Regression (Ridge when r!=None, else Kernel Regression) using
an RBF kernel, here a Gaussian one. We use pwtools.rbf.hyperopt.fit_opt() which
optimizes p and r by minimization of a CV MSE fit error, using a global
optimizer (scipy.optimize.differential_evolution()). Note that r is a measure
of the global noise level (usually called λ in KRR).

sklearn.gaussian_process.GaussianProcessRegressor
-------------------------------------------------
sklearn.gaussian_process.kernels.RBF kernel is the same as
pwtools.rbf.rbf_gauss(), i.e. the only radial basis function kernel in sklearn
is the Gaussian RBF (a.k.a. squared-exponential kernel). Additionally we use a
WhiteKernel to learn global noise, so the kernel we use is actually a
combination of two kernels which are responsible for modeling different aspects
of the data (i.e. "kernel engineering"). The GP implementation optimizes the
kernel's params (RBF(length_scale=...) and WhiteKernel(noise_level=...))
directly while fitting the model.

TODO: it would be cool to learn the per-point noise, that would be much more
useful.

GP optimizer
------------

We define a custom GP optimizer using scipy.optimize.differential_evolution().
The default optimizer, also with n_restarts_optimizer > 0 produces rubbish
results because it gets stuck in local optima time and time again and optimized
kernel params are useless.

Example results of optimized models
-----------------------------------

GP:

k1 is the Gaussian RBF kernel. length_scale (same as Rbf.p) is the optimized
kernel width parameter. k2 is the WhiteKernel with its optimized noise_level
parameter. This is equivalent to *but not the same as* Rbf.r . This would be
GaussianProcessRegressor(alpha=...) which we set to zero explicitly in order to
model the noise only by WhiteKernel. The *_bounds are printed because we set
them (else they'd have some other default value) but they are actually not used
unless n_restarts_optimizer > 0, which we don't do.

{'k1': RBF(length_scale=0.151),
 'k1__length_scale': 0.15135430570933753,
 'k1__length_scale_bounds': (1e-05, 100),
 'k2': WhiteKernel(noise_level=0.158),
 'k2__noise_level': 0.1580415180415214,
 'k2__noise_level_bounds': (1e-18, 1)}

Rbf:

{'ndim': 1,
 'p': 0.6238865254120896,
 'r': 1.0649344258412956e-06,
 'rbf': <function rbf_gauss at 0x7f074f1cb430>}

See also
--------
https://scikit-learn.org/stable/modules/gaussian_process.html#comparison-of-gpr-and-kernel-ridge-regression
"""

from pprint import pprint

import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from pwtools.rbf import estimate_p
import pwtools.rbf.hyperopt as rbf_hyperopt


def gt_func(x):
    """Ground truth"""
    return np.sin(x) * np.exp(-0.1 * x) + 10


def noise(x):
    return np.random.randn(x.shape[0]) / 10


def gp_optimizer(obj_func, initial_theta, bounds):
    """Custom optimizer for GaussianProcessRegressor using
    differential_evolution.
    """
    # Avoid pickle error when using multiprocessing in
    # differential_evolution(..., workers=-1).
    global obj_func_wrapper

    def obj_func_wrapper(theta):
        ##print(f"{obj_func(initial_theta)=}")
        # obj_func(theta, eval_gradient=True) hard-coded in
        # GaussianProcessRegressor, so it always returns the function value and
        # grad. However, we only need the function's value
        # differential_evolution() below.
        val, grad = obj_func(theta)
        return val

    opt_result = so.differential_evolution(
        obj_func_wrapper,
        bounds=bounds,
        **de_kwds_common,
    )
    print(f"gp_optimizer: {opt_result=}")
    return opt_result.x, opt_result.fun


def transform_1d(scaler, arr1d):
    assert arr1d.ndim == 1
    return scaler.transform(arr1d.reshape(-1, 1))[:, 0]


def de_callback(xk, convergence=None):
    """Callback for differential_evolution that prints the best individual per
    iteration."""
    print(f"{xk=} {convergence=}")


if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)

    x = np.linspace(0, 30, 50)
    xspan = x.max() - x.min()
    xi = np.linspace(x.min() - 0.3*xspan, x.max() + 0.3*xspan, len(x) * 10)
    y = gt_func(x) + noise(x)
    yi_gt = gt_func(xi)

    # For both rbf and gp below, make sure we work with the same scaled data.
    in_scaler = StandardScaler().fit(x.reshape(-1, 1))
    out_scaler = StandardScaler().fit(y.reshape(-1, 1))
    x = transform_1d(in_scaler, x)
    xi = transform_1d(in_scaler, xi)
    y = transform_1d(out_scaler, y)
    yi_gt = transform_1d(out_scaler, yi_gt)
    X = x[:, None]
    XI = xi[:, None]

    rbf_p_bounds = (1e-5, 100)
    rbf_p_init = estimate_p(X)

    # common settings for differential_evolution
    de_kwds_common = dict(
        polish=False,
        disp=False,
        tol=0.01,
        popsize=30,
        maxiter=10000,
        workers=-1,
        updating="deferred",
        seed=seed,
        callback=de_callback,
    )

    print("opt rbf ...")
    f_rbf = rbf_hyperopt.fit_opt(
        X,
        y,
        what="pr",
        ##what="p",
        rbf_kwds=dict(rbf="gauss"),
        cv_kwds=dict(random_state=123, n_splits=5, n_repeats=1),
        opt_kwds=dict(
            bounds=[rbf_p_bounds, (1e-18, 1e-5)],
            ##bounds=[rbf_p_bounds],
            **de_kwds_common,
        ),
    )

    gp_kernel = RBF(
        length_scale=rbf_p_init, length_scale_bounds=rbf_p_bounds
    ) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-18, 1))

    # n_restarts_optimizer > 0 produces identical runs of
    # differential_evolution, so skip it.
    print("opt gp ...")
    f_gp = GaussianProcessRegressor(
        kernel=gp_kernel,
        n_restarts_optimizer=0,
        optimizer=gp_optimizer,
        normalize_y=True,
        alpha=0,
    ).fit(X, y)

    # optimized kernel params
    pprint(f_gp.kernel_.get_params(deep=True))

    # optimized rbf params p and r
    pprint(f_rbf)

    # Sanity check: Use GaussianProcessRegressor with optimized f_rbf params
    # where alpha=Rbf.r = Ridge Regression λ, RBF(length_scale=Rbf.p) and don't
    # optimize them (optimizer=None). Also use normalize_y=False as we don't do
    # that in Rbf either. Must produce exactly the same result as Rbf(...,
    # rbf="gauss", r=f_rbf.r, p=f_rbf.p).
    f_gp_fixed = GaussianProcessRegressor(
        kernel=RBF(length_scale=f_rbf.p),
        optimizer=None,
        normalize_y=False,
        alpha=f_rbf.r,
    ).fit(X, y)
    # predictions
    assert np.allclose(f_gp_fixed.predict(XI), f_rbf(XI))
    # weights
    assert np.allclose(f_gp_fixed.alpha_, f_rbf.w)

    yi_rbf = f_rbf(XI)
    yi_gp, yi_gp_std = f_gp.predict(XI, return_std=True)
    ##yi_gp, yi_gp_std = f_gp_fixed.predict(XI, return_std=True)
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(x, y, "o", color="tab:gray", alpha=0.5)
    axs[0].plot(xi, yi_rbf, label="rbf", color="tab:red")
    axs[0].plot(xi, yi_gp, label="gp", color="tab:green")
    axs[0].fill_between(
        xi,
        yi_gp - 2 * yi_gp_std,
        yi_gp + 2 * yi_gp_std,
        alpha=0.2,
        color="tab:green",
        label="gp std",
    )

    yspan = y.max() - y.min()
    axs[0].plot(xi, yi_gt, label="gt", color="tab:gray", alpha=0.5)
    axs[0].set_ylim(y.min() - 0.1*yspan, y.max() + 0.1*yspan)
    axs[0].legend()

    diff_rbf = yi_gt - yi_rbf
    diff_gp = yi_gt - yi_gp
    msk = (xi > x.min()) & (xi < x.max())
    lo = min(diff_rbf[msk].min(), diff_gp[msk].min())
    hi = max(diff_rbf[msk].max(), diff_gp[msk].max())
    span = hi - lo
    axs[1].plot(xi, diff_rbf, label="rbf - gt", color="tab:red")
    axs[1].plot(xi, diff_gp, label="gp - gt", color="tab:green")
    axs[1].legend()
    axs[1].set_ylim((lo - 0.1*span, hi + 0.1*span))

    plt.show()
