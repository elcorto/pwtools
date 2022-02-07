#!/usr/bin/python3

"""
Study of cross-validation parameters. We use repeated K-fold cross-validation
and look at the CV mean, median and std as a function of n_splits, n_repeats
and number of data points.

While there is some research such as https://arxiv.org/abs/1811.12808
investigating the effect of some of those setting, there seems to be no simple
metric against which one can converge them and so people to just pick K=5 or
10. To at least get some intuition, and with cv the resulting 1d array of length
(n_splits*n_repeats) of fit errors from CV, we define a "converged CV" as (i)
mean(cv) = median(cv) (lacking any statistical justification, mind you!) and (ii)
mean(cv) is nearly independent of n_splits and n_repeats

When doing that, find that we need 200 data points or more (npoints). 100 is
kind of OK, but lower produces unreliable errors. 5-fold CV is enough
(n_splits=5) in most cases. More splits (n_splits>5) tend to lower the median
error for low npoints=50, when however the error itself is not yet converged.
For npoints>=200, there is no effect. More repeats (n_repeats>1) do not change
things, either.

We evaluate the CV error at RBF width parameter p=3, which we know results
sizable CV error values.
"""

import numpy as np
from pwtools import rbf
from pwtools.rbf.hyperopt import FitError

if __name__ == "__main__":
    rnd = np.random.RandomState(seed=1234)
    ##for rbf_type in ["gauss", "multi", "inv_multi"]:
    for rbf_type in ["gauss"]:
        print(rbf_type)
        print("npoints n_repeats n_splits mean median std")
        rbf_kwds = dict(rbf=rbf_type, r=1e-11)
        for npoints in [50, 100, 200, 300, 400]:
            x = np.linspace(0, 10, npoints)
            y = np.sin(x) + rnd.randn(npoints)
            print("")
            for n_repeats in [1, 2]:
                for n_splits in [5, 10, 20]:
                    fe = FitError(
                        x[:, None],
                        y,
                        cv_kwds=dict(
                            n_splits=n_splits,
                            n_repeats=n_repeats,
                            random_state=1234,
                        ),
                        rbf_kwds=rbf_kwds,
                    )
                    cv = fe.cv([3.0]) * 100
                    mean = np.mean(cv)
                    median = np.median(cv)
                    std = np.std(cv)
                    print(
                        f"{npoints:3} {n_repeats:3} {n_splits:3} {mean:12.1f} {median:12.1f} {std:12.1f}"
                    )


# Behavior when the model fit is always the ground truth.

##import numpy as np
##from sklearn.model_selection import RepeatedKFold
##
##if __name__ == "__main__":
##    rnd = np.random.RandomState(seed=1234)
##    print("npoints n_repeats n_splits mean median std")
##    for npoints in [50, 100, 200, 300, 400]:
##        x = np.linspace(0, 10, npoints)
##        y = np.sin(x) + rnd.randn(npoints)
##        print("")
##        for n_repeats in [1, 2]:
##            for n_splits in [5, 10, 20]:
##                cv = RepeatedKFold(
##                    n_splits=n_splits, n_repeats=n_repeats, random_state=rnd
##                )
##                errs = np.empty((cv.get_n_splits(x),), dtype=float)
##                for ii, tup in enumerate(cv.split(x)):
##                    idxs_train, idxs_test = tup
##                    d = np.sin(x[idxs_test]) - y[idxs_test]
##                    errs[ii] = np.dot(d, d)
##                mean = np.mean(errs)
##                median = np.median(errs)
##                std = np.std(errs)
##                print(
##                    f"{npoints:3} {n_repeats:3} {n_splits:3} {mean:12.1f} {median:12.1f} {std:12.1f}"
##                )
