#!/usr/bin/python3

"""
Map the cross-validation error in p-r space.
--------------------------------------------

p = RBF free parameter, width of the function
r = regularization strength

Problem
-------
Use a 1D fit problem (noisy sine) with large noise where interpolation of all
points will have large CV error, i.e. CV favors smooth fits instead. See
overfit.py

Observations
------------
We find that in regions of high r (strong regularization = strong smoothing)
where r is around 0.1 (log10(r)=-1), we can apply rather low p (narrow RBFs
which can overfit) b/c r provides enough stiffness.

In regions of typical low r (1e-6 and lower), it basically doesn't matter which
p we use, as long as it is big enough to not overfit. In these regions, both p
and r provide stiffness (also called "good generalization" in ML speak).

Global optimization
-------------------
We also try to calculate the global optimal (p,r) with
scipy.optimize.differential_evolution() and scipy.optimize.brute(). But the
large p-r valley of low CV error is flat and a bit rugged such that there is no
pronounced global optimum [*]. This supports our experimental experience that
the p value is basically irrelevant as long as it is not too small. B/c of the
flat valley, there is no use for optimization. One simply needs to know the
error landscape.

[*] Even though we always tend to land in the upper left corner around p=1, r=1
with differential_evolution, but we're as of now not sure whether this is a
real global opt of the flat valley or some artifact of the optimizer (or it's
parameters, not enough steps?). Anyway, we get a similar global opt from
brute() when we use really converged CV settings, i.e. npoints > 300. For lower
npoints, the brute global opt is at p ~ 10 and r=1e-12 (lowest possible value).
But again, setting zmax to 0.1 (for plotting the p-r map), we see that the
"flat" p-r valley of low CV error is very rugged and there really isn't a
global opt. Also, in this landscape, every optimizer will give you basically a
random result.

p-r map of different RBFs
-------------------------
gauss and inv_multi have virtually identical behavior, while for multi we find
a very inconsistent landscape with many error spikes in the valley. This is not
fully understood yet since the CV error convergence is the same for multi and
inv_multi (but this is evaluated at one p value only). The only difference
between all of them is that gauss and inv_multi have similar shape
characteristics (bell-like curves with tails approaching zero). multi is
inverted (but that's only a question of the sign), but has no such tails. It
doesn't get narrow for low p as gauss and inv_multi but instead approaches a
triangle shape, which may not be optimal for fitting continuous functions.
"""

import itertools
import multiprocessing

from matplotlib import colors
import numpy as np

from pwtools import rbf, mpl
from pwtools.rbf.hyperopt import FitError, fit_opt


plt = mpl.plt
plt.rcParams["figure.autolayout"] = True
color_cycler = iter(colors.TABLEAU_COLORS.values())
##export = True
export = False


if export:
    plt.rcParams["font.size"] = 10


def get_cv_kwds(seed):
    return dict(
        n_splits=5, n_repeats=1, random_state=np.random.RandomState(seed=seed)
    )


if __name__ == "__main__":
    # cut off errors bigger than zmax
    zmax = 0.3

    # random seed
    seed = 1234

    # Since the CV error uses random splits, it is important to pass this thru
    # to all FitError instances and fit_opt() calls if repeated runs shall
    # generate the same results.
    #
    # But still: Unless really converged CV settings are used (most important:
    # npoints, see crossval_convergence.py and below), you may find that
    # sometimes (repeated runs of this script) the global opt from fit_opt()
    # and the plotted p-r map calculated earlier don't fully match up. For
    # instance it may happen that the opt is outside the region of the flat p-r
    # valley of low CV error. This could also be misinterpreted as a failure of
    # the optimizer in the differential_evolution case, but surely not in case
    # of a supper simple solid method such as brute().
    rnd = np.random.RandomState(seed=seed)

    # nsample: nsample x nsample points in p-r space
    #
    # npoints: Number of data points of the fit problem. We use 100 for more
    # speed in this demo even though we know that the CV error starts to really
    # converge for more points. Still, 100 already gives quite OK error maps
    # which show the essential features. See crossval_convergence.py .
    if export:
        nsample = 100
        # converged CV:
        #   gauss: npoints >= 200
        #   multi, inv_multi: npoints >= 400
        npoints = 200
    else:
        nsample = 20
        npoints = 80

    rbf_names = ["gauss", "multi", "inv_multi"]
    ##rbf_names = ['gauss']
    plots = mpl.prepare_plots(rbf_names + ["fit"])
    x = np.linspace(0, 10, npoints)
    y = np.sin(x) + rnd.rand(len(x))
    plots["fit"].ax.plot(x, y, "o", label="data", color=next(color_cycler))
    p = np.linspace(0.01, 15, nsample)
    r = np.logspace(-12, 1, nsample)
    grid = np.array(list(itertools.product(p, r)))
    for name in rbf_names:
        rbf_kwds = dict(rbf=name)
        print(name)
        fig, ax = plots[name].fig, plots[name].ax
        ax.set_title(name)
        fe = FitError(
            x[:, None], y, cv_kwds=get_cv_kwds(seed), rbf_kwds=dict(rbf=name)
        )
        print("p-r map ...")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            zz = np.array(pool.map(fe, grid))
        ##zz[zz > zmax] = zmax
        zz[zz > zmax] = np.nan
        dd = mpl.Data2D(xx=grid[:, 0], yy=np.log10(grid[:, 1]), zz=zz)
        pl = ax.contourf(dd.X, dd.Y, dd.Z, levels=80)
        fig.colorbar(pl)
        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$\log_{10}(r)$")

        if export:
            for ext in ["png"]:
                fig.tight_layout()
                fig.savefig(f"/tmp/crossval_pr_{name}.{ext}")

        print("global opts ...")
        f_de = fit_opt(
            x[:, None],
            y,
            method="de",
            what="pr",
            opt_kwds=dict(
                disp=True,
                maxiter=30,
                popsize=50,
                bounds=[(p.min(), p.max()), (r.min(), r.max())],
                polish=False,
                updating="deferred",  # if workers=-1
                workers=-1,
                seed=seed,
            ),
            rbf_kwds=rbf_kwds,
            cv_kwds=get_cv_kwds(seed),
        )
        f_brute = fit_opt(
            x[:, None],
            y,
            method="brute",
            what="pr",
            opt_kwds=dict(
                disp=True,
                ranges=[(p.min(), p.max()), (r.min(), r.max())],
                finish=None,
                Ns=30,
                workers=-1,
            ),
            rbf_kwds=rbf_kwds,
            cv_kwds=get_cv_kwds(seed),
        )

        color = next(color_cycler)
        for opt_name, f_opt, ls in [
            ("brute", f_brute, "--"),
            ("de", f_de, "-"),
        ]:
            popt = f_opt.get_params()
            print(f"  {opt_name}: popt={popt}")
            ax.text(
                popt[0],
                np.log10(popt[1]),
                opt_name,
                va="center",
                ha="center",
                bbox=dict(fc="w", ec="k"),
            )

            xx = np.linspace(x[0], x[-1], 5 * len(x))
            plots["fit"].ax.plot(
                xx,
                f_opt(xx[:, None]),
                color=color,
                ls=ls,
                label=f"{name}-{opt_name}",
            )

    plots["fit"].legend()

    for pl in plots.values():
        pl.fig.tight_layout()
    mpl.plt.show()
