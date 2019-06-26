#!/usr/bin/python3

"""
Convergence of the cross-validation error. Use repeated K-fold cross-validation
(ns folds (splits), nr repititions). Find the minimum number of data points and
the best CV parameters for further experiments.

We define converged as mean(cv) = median(cv) where cv is the resulting 1d array
of length (ns*nr) of fit errors from CV.

We find that we need 200 data points or more (npoints). 100 is kind of OK, but
lower produces unreliable errors. 5-fold CV is enough (ns=5) in most cases.
More splits (ns>5) tend to lower the median error for low npoints=50, when
however the error itself is not yet converged. For npoints>=200, there is no
effect. More repeats (nr>1) do not increase the error quality at all.

Since sklearn's RepeatedKFold uses random splits, the results of this
experiment here are a bit different each time (in theory, we would need to
repeat and average that as well :). Most of the time, the gauss RBF behaves the
best and often converges for npoints=100 already, while multi and inv_multi
don't.

We evaluate the CV error at RBF width parameter p=3, which we know results
sizable CV error values.
"""

import numpy as np
from pwtools import rbf

if __name__ == '__main__':
    rnd = np.random.RandomState(seed=1234)
    for rbf_type in ['gauss', 'multi', 'inv_multi']:
        print(rbf_type)
        print("npoints nr ns mean median")
        rbf_kwds = dict(rbf=rbf_type)
        for npoints in [50, 100, 200, 400]:
            x = np.linspace(0, 10, npoints)
            y = np.sin(x) + rnd.rand(npoints)
            print("")
            for nr in [1, 2]:
                for ns in [5, 10, 20]:
                    fe = rbf.FitError(x[:,None], y, cv_kwds=dict(ns=ns, nr=nr),
                                      rbf_kwds=rbf_kwds)
                    cv = fe.cv([3.0])*100
                    mean = np.mean(cv)
                    median = np.median(cv)
                    print(f'{npoints:3} {nr:3} {ns:3} {mean:12.1f} {median:12.1f}')
