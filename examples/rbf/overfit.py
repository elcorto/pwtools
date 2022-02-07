#!/usr/bin/python3

"""
Show effect of different settings of the length scale parameter p in RBF
regression.

We use Rbf in Kernel Ridge Regression mode (r > 0 and fixed, so constant weak
smoothing) or least squares mode (r=None). In any case we vary only p.

There is a global minimum p_opt_mse of the loss (MSE cost/merit/loss/objective
function) at very small p (narrow RBFs can fit all points). It turns out that
this is almost exactly p_scipy in scipy.interpolate.Rbf, which is good for
interpolation, but not for regression.

Note that the loss landscape is pretty rugged and flat around p_pwtools (mean
distance estimate) such that a local optimizer such as 'fmin' won't produce
anything more meaningful. We use method='de' (differential_evolution) to make
sure we find the global minimum p_opt_mse for demonstration purpose.

The CV loss has a local minimum where the normal loss (MSE) has the global
minimum (p_opt_mse, p_scipy), but the CV loss's global minimum is a wide wange
between p ~ 1..20, where p_pwtools and p_opt_cv are located.

Note that we use a low number of points here for speed, such that the CV loss
is not yet converged, but shows the correct trend already.
"""

import numpy as np
from pwtools import mpl, rbf, batch
from pwtools.rbf.hyperopt import FitError, fit_opt

plt = mpl.plt


if __name__ == "__main__":
    seed = 123
    rnd = np.random.default_rng(seed=seed)
    cases = [batch.Case(r=None, name="lstsq"), batch.Case(r=1e-11, name="reg")]
    plots = mpl.prepare_plots([c.name for c in cases], nrows=2)

    x = np.linspace(0, 10, 100)
    y = np.sin(x) + rnd.random(len(x))
    points = x[:, None]
    values = y
    xi = np.linspace(x[0], x[-1], len(x) * 4)[:, None]

    cv_kwds = dict(n_splits=5, n_repeats=1, random_state=seed)

    for case in cases:
        rbf_kwds = dict(rbf="gauss", r=case.r)

        fe = FitError(points, values, rbf_kwds=rbf_kwds, cv=cv_kwds)

        fig = plots[case.name].fig
        ax_top = plots[case.name].ax[0]
        ax_top.set_title(f"r={case.r}")
        ax_top.plot(x, y, "o", alpha=0.3)

        rbfi = rbf.Rbf(points, values, **rbf_kwds)
        p_pwtools = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "r", label="$p$ pwtools")

        # cv=None: Use normal MSE loss, not CV loss
        rbfi = fit_opt(
            points,
            values,
            method="de",
            what="p",
            opt_kwds=dict(bounds=[(0.1, 20)], maxiter=10),
            rbf_kwds=rbf_kwds,
            cv=None,
        )
        p_opt_mse = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "g", label="$p$ opt mse")

        rbfi = fit_opt(
            points,
            values,
            method="de",
            what="p",
            opt_kwds=dict(bounds=[(0.1, 20)], maxiter=10),
            rbf_kwds=rbf_kwds,
            cv=cv_kwds,
        )
        p_opt_cv = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "y", label="$p$ opt cv")

        rbfi = rbf.Rbf(
            points, values, p=rbf.estimate_p(points, "scipy"), **rbf_kwds
        )
        p_scipy = rbfi.get_params()[0]
        ax_top.plot(xi, rbfi(xi), "m", label="$p$ scipy")

        ax_top.set_xlabel("x")
        ax_top.set_ylabel("y")
        ax_top.legend()

        loss_col = "tab:blue"
        cv_loss_col = "tab:orange"
        ax_bot_mse = plots[case.name].ax[1]
        ax_bot_cv = ax_bot_mse.twinx()
        p_range = np.logspace(np.log10(1e-2), np.log10(100), 100)
        ax_bot_mse.semilogx(
            p_range,
            [fe.err_direct([px]) for px in p_range],
            label="loss",
            color=loss_col,
        )
        ax_bot_cv.loglog(
            p_range,
            [fe.err_cv([px]) for px in p_range],
            cv_loss_col,
            label="CV loss",
        )
        ax_bot_mse.semilogx(
            [p_pwtools], fe.err_direct([p_pwtools]), "ro", label="$p$ pwtools"
        )
        ax_bot_mse.semilogx(
            [p_scipy], fe.err_direct([p_scipy]), "mo", label="$p$ scipy"
        )
        ax_bot_mse.semilogx(
            [p_opt_mse], fe.err_direct([p_opt_mse]), "go", label="$p$ opt mse"
        )
        ax_bot_cv.loglog(
            [p_opt_cv], fe.err_cv([p_opt_cv]), "yo", label="$p$ opt cv"
        )
        ax_bot_mse.set_xlabel("$p$")
        ax_bot_mse.set_ylabel("loss")
        ax_bot_cv.set_ylabel("CV loss")
        ax_bot_cv.legend(
            *mpl.collect_legends(ax_bot_mse, ax_bot_cv), loc="center right"
        )

        mpl.color_ax(ax_bot_mse, color=loss_col, axis="y", spine_loc="left")
        mpl.color_ax(
            ax_bot_cv,
            color=cv_loss_col,
            axis="y",
            spine_loc="right",
            spine_invisible="left",
        )

        fig.tight_layout()
        ##fig.savefig(f"/tmp/overfit_{case.name}.png")

    plt.show()
