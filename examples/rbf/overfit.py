#!/usr/bin/python3

"""
Pathological case of overfitting (interpolation of all points) when p is chosen
too small.

We use Rbf in Kernel Ridge Regression mode (r > 0 and fixed, so constant weak
smoothing) or least squares mode (r=None). In any case we vary only p.

There is a global minimum p_opt of the fit error at very small p (narrow RBFs
can fit all points). It turns out that this is almost exactly p_scipy in
scipy.interpolate.Rbf, which is good for interpolation, but not for fitting.

Note that the fit error landscape is pretty rugged and flat around the p_pwtools
(mean distance estimate) such that a local optimizer such as 'fmin' won't
produce anything more meaningful. We use method='de' (differential_evolution)
to make sure we find the global minimum p_opt for demonstration purpose.

This example shows that it is paramount to use cross-validation to calculate a
more useful fit error estimate. The CV error has a local minimum whete the
normal fit error (MSE) has the global minimum (p_opt, p_scipy), but the CV fit
error's global minimum is a wide wange between p ~ 1..20, where p_pwtools and
p_big are located.

Note that we use a low number of points here for speed, such that the CV error
is not yet converged, but shows the correct trend already.
"""

import numpy as np
from pwtools import mpl, rbf, batch
from pwtools.rbf.hyperopt import FitError, fit_opt

plt = mpl.plt


if __name__ == "__main__":
    rnd = np.random.RandomState(seed=123)
    cases = [batch.Case(r=None, name="lstsq"), batch.Case(r=1e-11, name="krr")]
    plots = mpl.prepare_plots([c.name for c in cases], nrows=2)
    for case in cases:
        rbf_kwds = dict(rbf="gauss", r=case.r)

        x = np.linspace(0, 10, 100)
        y = np.sin(x) + rnd.rand(len(x))
        points = x[:, None]
        values = y
        xi = np.linspace(x[0], x[-1], len(x) * 4)[:, None]

        fe = FitError(points, values, rbf_kwds=rbf_kwds)

        fig = plots[case.name].fig
        ax_top = plots[case.name].ax[0]
        ax_top.set_title(f"{case.name}: r={case.r}")
        ax_top.plot(x, y, "o", alpha=0.3)

        rbfi = rbf.Rbf(points, values, **rbf_kwds)
        p_pwtools = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "r", label="$p$ pwtools")

        # cv_kwds=None: Use normal MSE fit error, not CV fit error
        rbfi = fit_opt(
            points,
            values,
            method="de",
            what="p",
            opt_kwds=dict(bounds=[(0.1, 20)], maxiter=10),
            rbf_kwds=rbf_kwds,
            cv_kwds=None,
        )
        p_opt = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "g", label="$p$ opt")

        rbfi = rbf.Rbf(points, values, p=10, **rbf_kwds)
        p_big = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "y", label="$p$ big")

        rbfi = rbf.Rbf(
            points, values, p=rbf.estimate_p(points, "scipy"), **rbf_kwds
        )
        p_scipy = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "m", label="$p$ scipy")

        ax_top.set_xlabel("x")
        ax_top.set_ylabel("y")
        ax_top.legend()

        ax_bot = plots[case.name].ax[1]
        ax_bot2 = ax_bot.twinx()
        p_range = np.logspace(np.log10(1e-2), np.log10(100), 100)
        ax_bot.semilogx(
            p_range, [fe.err_direct([px]) for px in p_range], label="fit error"
        )
        ax_bot2.loglog(
            p_range,
            [fe.err_cv([px]) for px in p_range],
            "k",
            label="CV fit error",
        )
        ax_bot.semilogx(
            [p_pwtools], fe.err_direct([p_pwtools]), "ro", label="$p$ pwtools"
        )
        ax_bot.semilogx(
            [p_scipy], fe.err_direct([p_scipy]), "mo", label="$p$ scipy"
        )
        ax_bot.semilogx([p_opt], fe.err_direct([p_opt]), "go", label="$p$ opt")
        ax_bot.semilogx([p_big], fe.err_direct([p_big]), "yo", label="$p$ big")
        ax_bot.set_xlabel("$p$")
        ax_bot.set_ylabel("fit error")
        ax_bot2.set_ylabel("CV fit error")
        ax_bot.legend(
            *mpl.collect_legends(ax_bot, ax_bot2), loc="center right"
        )

        fig.tight_layout()

    plt.show()
##    fig.savefig("/tmp/overfit.png")
