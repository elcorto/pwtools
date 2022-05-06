.. include:: ../refs.rst

.. _rbf:


Radial Basis Function interpolation an regression
=================================================

Some background information on the method implemented in :mod:`~pwtools.rbf`.
For code examples, see the doc string of :class:`~pwtools.rbf.core.Rbf` and
``examples/rbf``.

Theory
------

The goal is to interpolate or fit an unordered set of :math:`M` ND data points
:math:`\mathbf x_i` and values :math:`z_i` so as to obtain :math:`z=f(\mathbf
x)`. In radial basis function (RBF) interpolation, the interpolating function
:math:`f(\mathbf x)` is a linear combination of RBFs :math:`\phi(r)`

.. math::
    f(\mathbf x) = \sum_j w_j\,\phi(|\mathbf x - \mathbf c_j|)

with the weights :math:`w_j` and the center points :math:`\mathbf c_j`.

An RBF :math:`\phi(r)` is a function of the distance :math:`r=|\mathbf x -
\mathbf c_j|` between :math:`\mathbf x` and a center point. Common functions
include

.. math::
    \begin{align}
        & \phi(r) = \exp\left(-\frac{r^2}{2\,p^2}\right) && \text{Gaussian}\\
        & \phi(r) = \sqrt{r^2 + p^2} && \text{multiquadric}\\
        & \phi(r) = \frac{1}{\sqrt{r^2 + p^2}} && \text{inverse multiquadric}
    \end{align}


All RBFs contain a single parameter :math:`p` which defines the length scale of the
function.

.. image:: ../../_static/rbfs.png

In interpolation the center points :math:`\mathbf c_j` are equal to the data points
:math:`\mathbf x_j` such that

.. math::
    \begin{gather}
        z_i = f(\mathbf x_i) = \sum_j w_j\,\phi(|\mathbf x_i - \mathbf x_j|) = \sum_j w_j\,K_{ij}\\
        \mathbf K\,\mathbf w = \mathbf z\\
    \end{gather}

with :math:`\mathbf K` the :math:`M\times M` matrix of RBF function values. The
weights :math:`\mathbf w = (w_j)` are found by solving the linear system
:math:`\mathbf K\,\mathbf w = \mathbf z`.

Note that for certain values of :math:`p`, :math:`\mathbf K` may become
singular, esp. large ones with the limit being
:math:`p\rightarrow\infty\Rightarrow\phi\rightarrow\text{const}`. Then one
should solve a regression problem instead (see below).


RBF parameter :math:`p`
-----------------------

Each RBF has a single "length scale" parameter :math:`p` (``Rbf(p=...)``,
attribute ``Rbf.p``). While :math:`f(\mathbf x)` goes through all points
(:math:`\mathbf x_i`, :math:`z_i`) by definition in interpolation, the behavior
between points is determined by :math:`p` where e.g. very narrow functions
:math:`\phi` can lead to oscillations between points. Therefore :math:`p` must
be optimized for the data set at hand, which should be done by cross-validation
as will be explained below.

Nevertheless we provide two methods to estimate reasonable start values.

The scipy implementation :math:`p_{\text{scipy}}` in
:class:`scipy.interpolate.Rbf` uses something like the mean nearest neighbor
distance. We provide this as ``Rbf(points, values, p='scipy')`` or
``rbf.estimate_p(points, 'scipy')``. The default in :class:`pwtools.rbf.Rbf`
however is the mean distance of all points

.. math::
   p_{\text{pwtools}}=1/M^2\,\sum_{ij} |\mathbf x_i - \mathbf x_j|

Use ``Rbf(points, values, p='mean')`` or ``rbf.estimate_p(points, 'mean')``.
This is always bigger than :math:`p_{\text{scipy}}` and can be a better start
guess for :math:`p`, especially in case of regression.


Interpolation vs. regression, regularization and numerical stability
--------------------------------------------------------------------

In case of noisy data, we may want to do regression. Similar to the `s`
parameter of :func:`scipy.interpolate.bisplrep` or the `smooth` parameter for
regularization in :class:`scipy.interpolate.Rbf`, we solve a regularized
version of the linear system

.. math::
        (\mathbf K + r\,\mathbf I)\,\mathbf w = \mathbf z

using, for instance, ``Rbf(points, values, r=1e-8)`` (attribute ``Rbf.r``). The
default however is :math:`r=0` (interpolation). :math:`r>0` creates a more
"stiff" (low curvature) function :math:`f(\mathbf x)` which does not
interpolate all points and where ``r`` is a measure of the assumed noise. If
the RBF happens to imply a Mercer kernel (which the Gaussian one does), then
this is equal to kernel ridge regression (KRR).

Apart from smoothing, the regularization also deals with the numerical
instability of :math:`\mathbf K\,\mathbf w = \mathbf z`.

One can also switch from :func:`scipy.linalg.solve` to
:func:`scipy.linalg.lstsq` and solve the system in a least squares sense
without regularization. You can do that by setting ``Rbf(..., r=None``). In
:class:`sklearn.kernel_ridge.KernelRidge` there is an automatic switch to least
squares when a singular :math:`\mathbf K` is detected. However, we observed
that in many cases this results in numerical noise in the solution, esp. when
:math:`\mathbf K` is (near) singular. So while we provide that feature for
experimentation, we actually recommend not using it in production and instead
use a KRR setup and properly determine :math:`p` and :math:`r` as described
below.

How to determine :math:`p` and :math:`r`
----------------------------------------

:math:`p` (and :math:`r`) should always be tuned to the data set at hand by
minimization of a cross validation (CV) score. You can do this with
:func:`pwtools.rbf.hyperopt.fit_opt`. Here is an example where we optimize only
:math:`p`, with :math:`r` constant and very small such as ``1e-11``. See
``examples/rbf/overfit.py``.


.. image:: ../../_static/overfit_reg.png

We observe a flat CV score above :math:`p>1` without a pronounced global
minimimum, even though we find a shallow one at around :math:`p=13`, while the
MSE loss (fit error) would suggest a very small :math:`p` value (i.e.
interpolation or "zero training loss" in ML terms).

Now we investigate :math:`p` and :math:`r` (see
``examples/rbf/crossval_map_p_r.py``) by plotting the CV score as function of
both. This is the result for the Gaussian RBF.

.. image:: ../../_static/crossval_pr_gauss.png

When :math:`r < 10^{-6}`, it basically doesn't matter which :math:`p` we use,
as long as it is big enough to not overfit, i.e. :math:`p>1`. In these regions,
both :math:`p` and :math:`r` provide stiffness. The large :math:`p`-:math:`r`
valley of low CV score is flat and a bit rugged such that there is no
pronounced global optimum, as shown above for :math:`r=10^{-11}`. As we incease
:math:`r` (stronger regularization = stronger smoothing) we are restricted to
lower :math:`p` (narrow RBFs which can overfit) since the now higher
:math:`r` already provides enough stiffness to prevent overfitting.

Data scaling
------------

In :class:`pwtools.num.PolyFit` (see the `scale` keyword) we scale
:math:`\mathbf x` and :math:`z` to :math:`[0,1]`. Here we don't have that
implemented but suggest to always scale at least :math:`z`. Since in contrast
to polynomials, the model here has no constant bias term, one should instead
normalize to zero mean using something like
:class:`sklearn.preprocessing.StandardScaler`.

Other implementations
---------------------

* :class:`scipy.interpolate.Rbf`
