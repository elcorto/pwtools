#!/usr/bin/env python3

"""
Compare hyper parameter optimization of pwtools KRR and sklearn GP
==================================================================

pwtools.rbf.Rbf(..., p=..., r=...)
----------------------------------

Kernel Ridge Regression (when r>0, (probably numerically unstable)
interpolation when r=0) using an RBF kernel function k, here a Gaussian one,
also called squared-exponential kernel. The kernel matrix is

    K_ij = k(x_i, x_j) = exp(-0.5 * d_ij^2 / p^2)
    d_ij = ||x_i - x_j||_2

In the KRR setting, we have an L2 regularization term in the loss (also called
"weight decay")

    r * ||w||_2^2

with the weights w, which results in a kernel matrix used in training where we
add r to the diagonal (L2 regularization variant of Tikhonov regularization)

    K_ii += r

We solve

    (K + r I)^-1 w = y

to get w.

We use pwtools.rbf.hyperopt.fit_opt() which optimizes p (kernel length scale
param) and r (KRR regularization param, also called λ) by minimization of a
cross-validation MSE fit error using a global optimizer
(scipy.optimize.differential_evolution()).


sklearn.gaussian_process.GaussianProcessRegressor
-------------------------------------------------

sklearn.gaussian_process.kernels.RBF is the same as pwtools.rbf.rbf_gauss(),
i.e. the only radial basis function kernel in sklearn is the Gaussian RBF.
Additionally we use a WhiteKernel "to learn global noise", so the kernel we use
is a combination of two kernels which are responsible for modeling different
aspects of the data (i.e. "kernel engineering"). The resulting kernel matrix is
the same as the above, i.e. RBF(length_scale=p)+WhiteKernel(noise_level=r) does

    K_ii += r

The only difference to KRR is that the GP implementation optimizes the kernel's
params (p,r) by maximization of the log marginal likelihood (LML) while KRR
needs to use CV, well and that we get y_std or y_cov if we want, so of course
the GP is in general the preferred solution. We haven't done any timings on large
data sets so we don't know if max LML or CV is faster. In any case, LML is the
mathematically more rigorous approach.

GP optimizer
------------

One can also specify r as GaussianProcessRegressor(alpha=r) if interpreted as
regularization parameter, in fact the default is not zero but 1e-10. However
the GP optimizer cannot optimize this, since it only optimizes kernel
hyperparameters, which is why we sneak it in via WhiteKernel(noise_level=r)
where we interpret it as noise, while setting alpha=0.

We define a custom GP optimizer using scipy.optimize.differential_evolution().
The default local optimizer (l_bfgs_b), also with n_restarts_optimizer > 0
doesn't find the global min in our test setup here. This is because the
LML(p,r) surface is tricky here.

Example results of optimized models
-----------------------------------

GP:

k1 is the Gaussian RBF kernel. length_scale (same as Rbf(p=...)) is the
optimized kernel width parameter. k2 is the WhiteKernel with its optimized
noise_level parameter (same as Rbf(r=...)).

{'k1': RBF(length_scale=0.147),
 'k1__length_scale': 0.14696558218508174,
 'k1__length_scale_bounds': (1e-05, 2),
 'k2': WhiteKernel(noise_level=0.0882),
 'k2__noise_level': 0.08820850820059796,
 'k2__noise_level_bounds': (0.001, 1)}

Fitted GP weights can also be accessed by
    GaussianProcessRegressor.alpha_
and optimized kernel hyper params by
    GaussianProcessRegressor.kernel_.k1.length_scale
    ...
(trailing underscores denote values after fitting (weights alpha_) and hyper
opt (kernel_)).

Rbf:

{'ndim': 1,
 'p': 0.11775436079522228,
 'r': 0.10582018438605084,
 'rbf': <function rbf_gauss at 0x7f2cd6f5dee0>}


The kernels, the noise and the log marginal likelihood
------------------------------------------------------

We use both models to solve

    (K + r I)^-1 w = y

and therefore the result of the hyperopt (p and r) should be the same ... which
it isn't.

The reason is that KRR (see also [1]) has to resort to something like CV to get
a useful optimization objective to find p and r, while GP can use maximization
of the LML. They can be equivalent, given one performs a very particular and
super costly variant of CV involving an "exhaustive leave-p-out
cross-validation averaged over all values of p and all held-out test sets when
using the log posterior predictive probability as the scoring rule", see
https://arxiv.org/abs/1905.08737 for details. This is nice but hard to do in
practice. Try to set p>1 in LeavePOut(p=...) and then wait... (note that we
ignore the "log posterior predictive scoring rule" part here, we just show that
LeavePOut is slow). Instead, we use KFold. This basically means that any form
of practically usable CV is an approximation of the LML with varying quality.

We also plot the CV and -LML surface as function of p and r. This shows
that, at least in the particular setting here (the data), those functions
are nasty specimens which are tricky to optimize using a local optimizer that
GaussianProcessRegressor uses by default. It also highlights how awesome
differential_evolution does the job, esp. given the fact that sklearn already
uses the log of p and r internally because (see
sklearn.gaussian_process.kernels.Kernel.theta, theta=[p,r] here):

    Note that theta are typically the log-transformed values of the
    kernel's hyperparameters as this representation of the search space
    is more amenable for hyperparameter search, as hyperparameters like
    length-scales naturally live on a log-scale.

We don't log() them in fit_opt() and still differential_evolution finds the
global min.

refs
----
[1] https://scikit-learn.org/stable/modules/gaussian_process.html#comparison-of-gpr-and-kernel-ridge-regression
"""

from pprint import pprint
import itertools
import multiprocessing as mp

import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeavePOut

from pwtools.rbf import Rbf, estimate_p
from pwtools.rbf.hyperopt import fit_opt, FitError
from pwtools import mpl

plt.rcParams["figure.autolayout"] = True


def gt_func(x):
    """Ground truth"""
    return np.sin(x) * np.exp(-0.1 * x) + 10


def noise(x, rng, noise_level=0.1):
    """Gaussian noise."""
    # variable noise
    ##return rng.normal(loc=0, scale=noise_level, size=x.shape[0]) * np.exp(0.1*x) / 5
    # constant noise
    return rng.normal(loc=0, scale=noise_level, size=x.shape[0])


def transform_1d(scaler, arr1d):
    assert arr1d.ndim == 1
    return scaler.transform(arr1d.reshape(-1, 1))[:, 0]


def de_callback_rbf(xk, convergence=None):
    """Callback for differential_evolution that prints the best individual per
    iteration."""
    print(f"rbf: {xk=}")


def de_callback_gp(xk, convergence=None):
    """Callback for differential_evolution that prints the best individual per
    iteration.

    Since GaussianProcessRegressor's hyper optimizer code path internally works
    with log(p) (= xk[0]) and log(r) (= xk[1]), we need to exp() them before
    printing."""
    print(f"gp: {np.exp(xk)=}")


def gp_optimizer(obj_func, initial_theta, bounds):
    """Custom optimizer for GaussianProcessRegressor using
    differential_evolution.

    Ignore initial_theta since we need only bounds for differential_evolution.
    """
    # Avoid pickle error when using multiprocessing in
    # differential_evolution(..., workers=-1).
    global obj_func_wrapper

    def obj_func_wrapper(theta):
        ##print(f"{obj_func(initial_theta)=}")
        # obj_func(theta, eval_gradient=True) hard-coded in
        # GaussianProcessRegressor, so it always returns the function value and
        # grad. However, we only need the function's value in
        # differential_evolution() below. obj_func = log_marginal_likelihood.
        val, grad = obj_func(theta)
        return val

    opt_result = so.differential_evolution(
        obj_func_wrapper,
        bounds=bounds,
        callback=de_callback_gp,
        **de_kwds_common,
    )
    print(f"gp_optimizer: {opt_result=}")
    return opt_result.x, opt_result.fun


if __name__ == "__main__":
    seed = 123
    rng = np.random.default_rng(seed=seed)

    # Equidistant x points: constant y_std (from GP) in-distribution
    ##x = np.linspace(0, 30, 60)
    #
    # Random x points for varying y_std. For some reason the results vary
    # depending on whether we sort the points. Shouldn't be the case?
    ##x = rng.uniform(0, 30, 60)
    x = np.sort(rng.uniform(0, 30, 60), axis=0)
    xspan = x.max() - x.min()
    xi = np.linspace(x.min() - 0.3 * xspan, x.max() + 0.3 * xspan, len(x) * 10)
    y = gt_func(x) + noise(x, rng, noise_level=0.1)
    yi_gt = gt_func(xi)

    # For both rbf and gp below, make sure we work with the same scaled data.
    # B/c we scale here, and for comparison with Rbf(), we use
    # GaussianProcessRegressor(normalize_y=False). In Rbf(), we don't have
    # normalization anyway and rely on feeding in normalized data.
    in_scaler = StandardScaler().fit(x.reshape(-1, 1))
    out_scaler = StandardScaler().fit(y.reshape(-1, 1))
    x = transform_1d(in_scaler, x)
    xi = transform_1d(in_scaler, xi)
    y = transform_1d(out_scaler, y)
    yi_gt = transform_1d(out_scaler, yi_gt)
    X = x[:, None]
    XI = xi[:, None]

    # Hyper-param bounds for optimization.
    param_p_bounds = (1e-5, 2)
    param_r_bounds = (1e-3, 1)

    # Common settings for differential_evolution. Since the mean CV and the LML
    # have different magnitudes, it would be an error to use the same atol
    # (absolute convergence tolerance). For that reason we use atol=0 (default
    # anyway) and use only relative convergence tolerance `tol`.
    de_kwds_common = dict(
        polish=False,
        disp=False,
        atol=0,
        tol=0.001,
        popsize=20,
        maxiter=10000,
        workers=-1,
        updating="deferred",
        seed=seed,
    )

    print("opt rbf ...")
    rbf_kwds = dict(rbf="gauss")
    rbf_cv = dict(random_state=seed, n_splits=5, n_repeats=1)
    ##rbf_cv = LeavePOut(p=1)
    f_rbf = fit_opt(
        X,
        y,
        method="de",
        what="pr",
        ##what="p",
        rbf_kwds=rbf_kwds,
        cv=rbf_cv,
        opt_kwds=dict(
            bounds=[param_p_bounds, param_r_bounds],
            ##bounds=[param_p_bounds],
            callback=de_callback_rbf,
            **de_kwds_common,
        ),
    )

    # Define GP kernel. Pass bounds for the optimizer used here. Pass scalar
    # start values if local optimizer is used.
    gp_kernel = RBF(
        ##length_scale=estimate_p(X),
        length_scale_bounds=param_p_bounds
    ) + WhiteKernel(
        ##noise_level=1,
        noise_level_bounds=param_r_bounds,
    )

    # n_restarts_optimizer > 0 produces identical runs of
    # differential_evolution, so skip it.
    print("opt gp ...")
    f_gp = GaussianProcessRegressor(
        kernel=gp_kernel,
        n_restarts_optimizer=0,
        optimizer=gp_optimizer,
        normalize_y=False,
        alpha=0,
    ).fit(X, y)

    # Print optimized kernel params
    pprint(f_gp.kernel_.get_params(deep=True))
    pprint(f_rbf)

    # Sanity check: expansion coeffs
    #   GaussianProcessRegressor.alpha_ (note the underscore!)
    #   Rbf.w
    #
    # Use GaussianProcessRegressor with optimized f_rbf params where param_r is
    # equal to
    #   GaussianProcessRegressor(alpha=...) OR WhiteKernel(noise_level=...)
    #   Rbf(r=...)
    #
    # and param_p is
    #   RBF(length_scale=...)
    #   Rbf(p=...)
    #
    # Don't optimize them (optimizer=None), just set to fixed values. Must
    # produce exactly the same result as Rbf(..., rbf="gauss", r=f_rbf.r,
    # p=f_rbf.p).
    f_gp_fixed_from_rbf = GaussianProcessRegressor(
        kernel=RBF(length_scale=f_rbf.p),
        optimizer=None,
        normalize_y=False,
        alpha=f_rbf.r,
    ).fit(X, y)
    # predictions
    np.testing.assert_allclose(f_gp_fixed_from_rbf.predict(XI), f_rbf(XI))
    # weights
    np.testing.assert_allclose(f_gp_fixed_from_rbf.alpha_, f_rbf.w)

    # Same as above, now use WhiteKernel(noise_level=...) instead of
    # GaussianProcessRegressor(alpha=...).
    f_gp_fixed_from_rbf = GaussianProcessRegressor(
        kernel=RBF(length_scale=f_rbf.p) + WhiteKernel(noise_level=f_rbf.r),
        optimizer=None,
        normalize_y=False,
        alpha=0,
    ).fit(X, y)
    # predictions
    np.testing.assert_allclose(f_gp_fixed_from_rbf.predict(XI), f_rbf(XI))
    # weights
    np.testing.assert_allclose(f_gp_fixed_from_rbf.alpha_, f_rbf.w)

    # Sanity check: noise
    #
    # We already covered that above with p and r from Rbf, but ... ok why not
    # again :)
    #
    # Verify that the learned global noise level from WhiteKernel is equal to
    # using this value as param_r without WhiteKernel, i.e. these are the
    # same:
    #   f_gp.kernel_.k2.noise_level   (WhiteKernel(noise_level=...) AFTER
    #                                  training, note the underscore in
    #                                  "kernel_"!!)
    #   GaussianProcessRegressor(alpha=...)
    #   Rbf(r=...)
    f_gp_fixed_from_gp = GaussianProcessRegressor(
        kernel=RBF(length_scale=f_gp.kernel_.k1.length_scale),
        optimizer=None,
        normalize_y=False,
        alpha=f_gp.kernel_.k2.noise_level,
    ).fit(X, y)
    f_rbf_fixed_from_gp = Rbf(
        X,
        y,
        p=f_gp.kernel_.k1.length_scale,
        r=f_gp.kernel_.k2.noise_level,
        rbf="gauss",
    )
    np.testing.assert_allclose(
        f_gp.predict(XI), f_gp_fixed_from_gp.predict(XI)
    )
    np.testing.assert_allclose(f_gp.predict(XI), f_rbf_fixed_from_gp(XI))

    # Sanity check: kernel matrix
    K_gp = f_gp.kernel_(X)
    ff = f_rbf_fixed_from_gp
    K_rbf = ff.rbf(ff.get_distsq(X), ff.p) + np.eye(X.shape[0]) * ff.r
    np.testing.assert_allclose(K_gp, K_rbf)

    # Plot functions, data and GP's std
    yi_rbf = f_rbf(XI)
    yi_gp, yi_gp_std = f_gp.predict(XI, return_std=True)

    fig1, axs = plt.subplots(
        nrows=3, sharex=True, gridspec_kw=dict(height_ratios=[1, 0.3, 0.3])
    )
    axs[0].plot(x, y, "o", color="tab:gray", alpha=0.5)
    axs[0].plot(xi, yi_rbf, label="rbf", color="tab:red")
    axs[0].plot(xi, yi_gp, label="gp", color="tab:green")
    axs[0].fill_between(
        xi,
        yi_gp - 2 * yi_gp_std,
        yi_gp + 2 * yi_gp_std,
        alpha=0.2,
        color="tab:gray",
        label=r"gp $\pm 2\,\sigma$",
    )

    yspan = y.max() - y.min()
    axs[0].plot(
        xi, yi_gt, label="ground truth g(x)", color="tab:gray", alpha=0.5
    )
    axs[0].set_ylim(y.min() - 0.1 * yspan, y.max() + 0.1 * yspan)

    diff_rbf = yi_gt - yi_rbf
    diff_gp = yi_gt - yi_gp
    msk = (xi > x.min()) & (xi < x.max())
    lo = min(diff_rbf[msk].min(), diff_gp[msk].min())
    hi = max(diff_rbf[msk].max(), diff_gp[msk].max())
    span = hi - lo
    axs[1].plot(xi, diff_rbf, label="rbf - g(x)", color="tab:red")
    axs[1].plot(xi, diff_gp, label="gp - g(x)", color="tab:green")
    axs[1].set_ylim((lo - 0.1 * span, hi + 0.1 * span))

    axs[2].plot(xi, yi_gp_std, label=r"gp $\sigma$")

    for ax in axs:
        ax.legend()

    # Plot hyperopt objective function. rbf: median CV fit error (see
    # FitError.err_cv()), gp: negative log_marginal_likelihood
    nsample = 50
    p = np.linspace(*param_p_bounds, nsample)
    r = np.linspace(*param_r_bounds, nsample)
    grid = np.array(list(itertools.product(p, r)))
    plots = mpl.prepare_plots(["rbf", "gp"])
    for name, plot in plots.items():
        print(name)
        fig, ax = plots[name].fig, plots[name].ax
        ax.set_title(name)
        if name == "rbf":
            zmax = 0.5
            func = FitError(X, y, cv=rbf_cv, rbf_kwds=rbf_kwds)
            pr_opt = f_rbf.get_params()
        elif name == "gp":
            zmax = 0.005

            def func(pr):
                return -f_gp.log_marginal_likelihood(np.log(pr))

            pr_opt = [
                f_gp.kernel_.k1.length_scale,
                f_gp.kernel_.k2.noise_level,
            ]
        with mp.Pool(mp.cpu_count()) as pool:
            zz = np.array(pool.map(func, grid))
        zz -= zz.min()
        zz /= zz.max()
        ##zz[zz > zmax] = zmax
        zz[zz > zmax] = np.nan
        dd = mpl.Data2D(XY=grid, zz=zz)
        pl = ax.contourf(dd.X, dd.Y, dd.Z, levels=80)
        fig.colorbar(pl)
        ax.plot(*pr_opt, "o", ms=10, color="white")
        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$r$")

    plt.show()
