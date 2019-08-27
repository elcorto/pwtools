#!/usr/bin/python3

"""
Map the cross-validation error in p-r space.

p = RBF free parameter, width of the function
r = regularization strength

Use a fit problem with large noise where interpolation of all points will have
large CV error, i.e. CV favors smooth fits instead. See overfit.py

We find that in regions of high r (strong regularization = strong smoothing)
where r is around 0.1 (log10(r)=-1), we can apply rather low p (narrow RBFs
which can overfit) b/c r provides enough stiffness.

In regions of typical low r (1e-6 and lower), it basically doesn't matter which
p we use, as long as it is big enough to not overfit. In these regions, both p
and r provide stiffness (also called "good generalization" in ML speak).

We also try to calculate the global optimal (p,r) with differential_evolution,
but the large p-r valley of good values is flat and a bit rugged such that
there is no pronounced global optimum [*]. This supports our experimental
experience that the p value is basically irrelevant as long as it is not too
small. B/c of the flat valley, there is no use for optimization. One simply
needs to know the error landscape.

[*] Even though we always land in the upper left corner around p=1, r=1 but
we're as of now not sure whether this is a real global opt of the flat valley
or some artifact of the differential_evolution.

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
from matplotlib import cm

import numpy as np
from pwtools import rbf, mpl
plt = mpl.plt

export = False

if export:
    savefig_opts = dict(bbox_inches='tight', pad_inches=0)
    plt.rcParams['font.size'] = 15


if __name__ == '__main__':
    # don't plot errors bigger than zmax
    zmax = 0.7

    # nsample: nsample x nsample points in p-r space
    # npoints: number of data points of the fit problem, we use 100 for more
    # speed in this demo even though we know that the CV error startes to
    # really converge for 200+ points, 100 already gives quite OK error maps
    # which show the essential features. See crossval_convergence.py .
    if export:
        nsample = 50
        npoints = 200
    else:
        nsample = 12
        npoints = 100

    rnd = np.random.RandomState(seed=1234)
    for name in ['gauss', 'multi', 'inv_multi']:
##    for name in ['gauss']:
        print(name)
        fig,ax = mpl.fig_ax()
        ax.set_title(name)
        x = np.linspace(0, 10, npoints)
        y = np.sin(x) + rnd.rand(len(x))
        fe = rbf.FitError(x[:,None], y, cv_kwds=dict(ns=5, nr=1),
                          rbf_kwds=dict(rbf=name))
        p = np.linspace(0.01, 15, nsample)
        r = np.logspace(-10, 2, nsample)
        g = np.array(list(itertools.product(p,r)))
        zz = np.array([fe.err_cv(params) for params in g])
        zz[zz > zmax] = zmax
        dd = mpl.Data2D(xx=g[:,0], yy=np.log10(g[:,1]), zz=zz)
        pl = ax.contourf(dd.X, dd.Y, dd.Z, cmap=cm.jet)
        plt.colorbar(pl)
        ax.set_xlabel(r'$p$')
        ax.set_ylabel(r'$\log_{10}(r)$')

        if export:
            for ext in ['png', 'pdf']:
                fig.savefig(f'/tmp/crossval_pr_{name}.{ext}',
                            **savefig_opts)
##        f = rbf.fit_opt(x[:,None], y, method='de', what='pr',
##                        opt_kwds=dict(disp=True,
##                                      maxiter=20,
##                                      popsize=20,
##                                      bounds=[(p.min(),p.max()),(r.min(),r.max())],
##                                      mutation=1.5,
##                                      polish=True),
##                        rbf_kwds=dict(rbf=name))
##        popt = f.get_params()
##        print(popt)
##        ax.plot([popt[0]], [popt[1]], 'ro')
    mpl.plt.show()
