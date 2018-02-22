.. _rbf:

Radial Basis Functions Networks for interpolation or fitting of N-dim data points
=================================================================================

Refs:

.. [1] http://en.wikipedia.org/wiki/Radial_basis_function_network
.. [2] Numerical Recipes, 3rd ed., ch 3.7

Training
--------
For our RBF network, we use the traditional approach and simply solve a linear
system for the weights. Calculating the distance matrix w/
``scipy.spatial.distance.cdist`` or :func:`~pwtools.num.distsq` is handled
efficiently, even for many points. But we get into trouble for many points
(order 1e4) b/c solving a big dense linear system (1e4 x 1e4) with plain
``scipy.linalg`` on a single core is possible but painful (takes some minutes)
-- the traditional RBF problem. Maybe use numpy build against threaded MKL, or
even scalapack? For the latter, look at how GPAW does this.

rbf parameter
-------------
Each RBF has a single parameter :math:`p`, which can be tuned. This is
usually a measure for the "width" of the function. e.g. in
:class:`~pwtools.rbf.RBFMultiquadric` :math:`\sqrt{r^2+p^2}`, attribute
``param`` in the code.

* What seems to work best is :class:`~pwtools.rbf.RBFMultiquadric` +
  ``RBFInt.get_param(param='est')`` (mean-distance of all points, default),
  i.e. the "traditional" RBF approach. This is exactly what's done in
  ``scipy.interpolate.Rbf``.

* It seems that for very few data points (say 5x5 for x**2 + x**2), the default
  mean-distance estimate is too small. We then have good interpolation at the
  data points, but wildly fluctuating bogus between them, since there is no
  data support in between the data points. We need to have wider RBFs. Usually,
  bigger (x10), sometimes much bigger (x100) params work.

* Similarly, in some cases params smaller than the mean-distance estimate
  provide lower fit error at the points, but with the same between-points behavior
  as above.

In general however, the mean-distance estimate is the best default one can use
(see below for more details).

Interpolation vs. fitting
-------------------------
For smooth noise-free data, RBF works perfect. But for noisy data, we would
like to do some kind of fit instead, like the "s" parameter to
``scipy.interpolate.bisplrep``. ``scipy.interpolate.Rbf`` has a "smooth"
parameter and what they do is some form of regularization (solve (``G-I*smooth)
. w = z`` instead of ``G . w = z``; ``G`` = RBF matrix, ``w`` = weights to
solve for, ``z`` = data).

We found (see ``examples/rbf.py``) that ``scipy.linalg.solve`` often gives an
ill-conditioned matrix warning, which shows numerical instability and results
in bad interpolation. It seems that problems start to show as soon as the noise
level (``z`` + noise) is in the same order or magnitude as the mean point distance.
Then we see wildly fluctuating data points which are hard to interpolate. In
that case, the mean-distance estimate for the rbf param breaks down and one
needs to use smaller values to interpolate all fluctuations. However, in most
cases, on does actually want to perform a fit instead in such situations.

If we switch from ``scipy.linalg.solve`` to ``scipy.linalg.lstsq`` and solve the
system in a least squares sense, we get much more stable solutions. With
``lstsq``, we have the smoothness by construction, b/c we do *not* perform
interpolation anymore -- this is a fit now. The advantage of using least
squares is that we don't have a smoothness parameter which needs to be tuned.

If the noise is low relative to the point distance, we get interpolation-like
results, which cannot be distinguished from the solutions obtained with a
normal linear system solver. The method will try its best to do interpolation,
but will smoothly transition to fitting as noise increases, which is what we
want. Hence, lstsq is the default solver.

optimize rbf param
------------------
With :meth:`~pwtools.rbf.RBFInt.fit_opt_param`, we try to optimize
``rbf.param`` by repeated fitting with ``fit(solver='lstsq')``. In all
experiments so far, we find that the mean-distance estimate of ``param`` is
actually very good and doesn't change much when optimizing. However, this is
not always the global min. There are cases where we find a much smaller
``param`` (e.g. factor 10 smaller), which leads to better fits at the data
points but oscillations between them (found with methods where we actually have
more points for testing than we use for fitting, see ``examples/rbf.py``). That
minimum was found sometimes with ``scipy.optimize.fmin`` and always with
``scipy.optimize.differential_evolution``. However, as stated above, we usually
want to do a fit with the mean-distance estimate rather than a perfect
interpolation in those cases.

other codes
-----------

* ``scipy.interpolate.Rbf``
  Essentially the same as we do. We took some ideas from there.
* http://pypi.python.org/pypi/PyRadbas/0.1.0
  Seems to work with Gaussians hardcoded. Not what we want.
* http://code.google.com/p/pyrbf/
  Seems pretty useful for large problems (many points), but not
  documented very much.

input data
----------
It usually helps if all data ranges (points X and values Y) are in the same
order of magnitude, e.g. all have a maximum close to unity or so. If, for
example, X and Y have very different scales (say X -0.1 .. 0.1 and Y
0...1e4), you may get bad interpolation between points. This is b/c the RBFs
also live on more or less equal x-y scales.
