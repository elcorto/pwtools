.. _rbf:

Radial Basis Functions Networks for interpolation or fitting of N-dim data points
=================================================================================

Some background information on the method implemented in :mod:`~pwtools.rbf`.
For code examples, see the doc string of :class:`~pwtools.rbf.Rbf`. See
``examples/rbf/`` for advanced topics such as cross-validation error
convergence, avoidance of overfitting, error landscape analysis and
optimization of the RBF parameter :math:`p` (and regularization strength
:math:`r`).

Refs:

.. [1] http://en.wikipedia.org/wiki/Radial_basis_function_network
.. [2] Numerical Recipes, 3rd ed., ch 3.7

Theory
------

The goal is to interpolate or fit an unordered set of :math:`M` ND data points
:math:`\mathbf x_i` and values :math:`z_i` so as to obtain :math:`z=f(\mathbf
x)`. In radial basis function (RBF) interpolation, the interpolating function
:math:`f(\mathbf x)` is a linear combination of RBFs :math:`\phi(r)`

.. math::
    f(\mathbf x) = \sum_j w_j\,\phi(|\mathbf x - \mathbf c_j|)

with the weights :math:`w_j` and the center points :math:`\mathbf c_j`. An RBF
:math:`\phi(r)` is thus a function of the distance :math:`r=|\mathbf x -
\mathbf c_j|` between :math:`\mathbf x` and a center point. Common functions
include

.. math::
    \begin{align}
        & \phi(r) = \exp\left(-\frac{r^2}{2\,p^2}\right) && \text{Gaussian}\\
        & \phi(r) = \sqrt{r^2 + p^2} && \text{multiquadric}\\
        & \phi(r) = \frac{1}{\sqrt{r^2 + p^2}} && \text{inverse multiquadric}
    \end{align}

All RBFs contain a single parameter :math:`p` which defines the width of the
function. The function :math:`f(\mathbf x)` can be also thought of as a neural
network with one hidden layer and activation functions :math:`\phi`.

In interpolation the center points :math:`\mathbf c_j` are equal to the data points 
:math:`\mathbf x_j` such that

.. math::
    \begin{gather}
        z_i = f(\mathbf x_i) = \sum_j w_j\,\phi(|\mathbf x_i - \mathbf x_j|) = \sum_j w_j\,G_{ij}\\
        \mathbf G\,\mathbf w = \mathbf z\\
    \end{gather}

with :math:`\mathbf G` the :math:`M\times M` matrix of RBF function values. The
weights :math:`\mathbf w = (w_j)` are found by solving the linear system
:math:`\mathbf G\,\mathbf w = \mathbf z`.

Thus, the applicability of the method is limited by the number of points
:math:`M` in the sense that a dense linear system :math:`M\times M` must be
stored and solved. For large point sets, the calculation of the distance matrix
:math:`R_{ij} = |\mathbf x_i - \mathbf x_j|` is one of the bottlenecks. In
pwtools, this is coded in Fortran (see :func:`~pwtools.num.distsq`). 

RBF parameter :math:`p`
-----------------------
Each RBF has a single "width" parameter :math:`p`, which can be tuned
(attribute ``Rbf.p`` in the code). While :math:`f(\mathbf x)` goes through all
data points :math:`\mathbf x_i` by definition (in interpolation, regularization
:math:`r\rightarrow 0`), the behavior of the interpolation between points is
determined by :math:`p`. For instance, too narrow functions :math:`\phi` can
lead to oscillations between points. Therefore :math:`p` must be tuned for the
specific data set. The scipy implementation :math:`p_{\text{scipy}}` in
``scipy.interpolate.Rbf`` uses something like the mean nearest neighbor
distance. We provide this as ``Rbf(points, values, p='scipy')`` or
``rbf.estimate_p(points, 'scipy')``. The default in :class:`pwtools.rbf.Rbf`
however is the mean distance of all points
:math:`p_{\text{pwtools}}=1/M^2\,\sum_{ij} R_{ij}` (``Rbf(points, values,
p='mean')`` or ``rbf.estimate_p(points, 'mean')``). This is always bigger than
:math:`p_{\text{scipy}}`, and yes this will change with the min-max span of the
data, while the mean nearest neighbor
distance stays constant. However it is usually the better
start guess for :math:`p` since it is less prone to overfitting in case of
noisy data, where the smaller scipy :math:`p` will often still interpolate all
points. However, there is no ad-hoc best choice for :math:`p`. In general,
:math:`p` must be determined by methods such as K-fold cross validation. Use
:class:`~pwtools.rbf.FitError`, esp. :meth:`~pwtools.rbf.FitError.err_cv` or
other tools from ``scikit-learn`` which are outside of the scope of pwtools.
See ``examples/rbf/overfit.py``.

Interpolation vs. fitting and regularization
--------------------------------------------
For smooth noise-free data, RBF provides nice interpolation. But for noisy
data, we would like to do a fit instead, similar to "s" parameter of
``scipy.interpolate.bisplrep``. ``scipy.interpolate.Rbf`` has a "smooth"
parameter for regularization. Here we can do the same (``Rbf.r``, ``Rbf(points,
values, r=1e-8)``) and solve

.. math::
        (\mathbf G + r\,\mathbf I)\,\mathbf w = \mathbf z

which creates a more "stiff" (low curvature) function :math:`f(\mathbf x)`
which does not necessarily interpolate all points. The regularization also
deals with the numerical instability of :math:`\mathbf G\,\mathbf w = \mathbf
z`, which results in ``scipy.linalg.solve`` often issuing an ill-conditioned
matrix warning and very bad interpolation results. 

See ``examples/rbf/`` for how to investigate the :math:`(p,r)` fit error
landscape and to calculate optimal :math:`p` and :math:`r` for
a given data set:

.. image:: ../../_static/crossval_pr_gauss.png

One can also switch from ``scipy.linalg.solve`` to ``scipy.linalg.lstsq`` and
solve the system in a least squares sense without regularization. In that case
we also get much more stable solutions. The advantage [*] of using least
squares is that we have the smoothness by construction no smoothness parameter
needs to be tuned. If the noise is low relative to the point distance, we get
interpolation-like results, which cannot be distinguished from the solutions
obtained with a normal linear system solver. The method will try its best to do
interpolation, but will smoothly transition to fitting as noise increases.

[*] However, we found that ``lstsq`` can introduce small numerical noise in the
    solution, so test before using (as always!).

Other codes
-----------

* ``scipy.interpolate.Rbf``
* http://pypi.python.org/pypi/PyRadbas/0.1.0
* http://code.google.com/p/pyrbf/

Input data
----------
It usually helps if all data ranges (points X and values Y) are in the same
order of magnitude, e.g. all have a maximum close to unity or so. If, for
example, X and Y have very different scales (say X -0.1 .. 0.1 and Y
0...1e4), you may get bad interpolation between points. This is b/c the RBFs
also live on more or less equal x-y scales.
