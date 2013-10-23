Coordinate transformation
=========================

Textbook relations and basic linalg
-----------------------------------

:math:`X`, :math:`Y` are square matrices with basis vecs as *columns*.

| :math:`X` ... old, shape: (3,3) .. or (M,M) in gerenal
| :math:`Y` ... new, shape: (3,3)
| :math:`I` ... identity matrix, basis vecs of cartesian system, shape: (3,3)
| :math:`A` ... transformation matrix, shape(3,3)
| :math:`v_X` ... column vector v in basis X, shape: (3,1)
| :math:`v_Y` ... column vector v in basis Y, shape: (3,1)
| :math:`v_I` ... column vector v in basis I, shape: (3,1)

We have

.. math::
    Y v_Y = X v_X = I v_I = v_I

    v_Y = Y^{-1} X v_X = A v_X

Note that :math:`(A B)^T = B^T A^T`, so for *row* vectors :math:`v^T`, we have

.. math::
    v_Y^T = (A v_X)^T = v_X^T A^T

Every product :math:`X v_X; Y v_Y; v_I I` is actually an expansion of
:math:`v_{X,Y,...}` in the basis vectors contained in :math:`X,Y,...` . If the
dot product is computed, we always get :math:`v` in cartesian coords. 

Now, :math:`v_X^T` is a row(!) vector (1,M). This form is implemented here (see
below for why). In numpy ``A.T`` is the transpose. ``v_X`` is actually an 1d array
for which ``v_X.T == v_X`` and therefore ``dot(A, v_X) == dot(v_X, A.T)``.

In general, we don't have one vector ``v_X`` but an array ``R_X`` of shape
(N,M) of row vectors::
    
    R_X = [[--- v_X0 ---],
           [--- v_X1 ---],
           ...
           [-- v_XN-1 --]]

We want to use fast numpy array broadcasting to transform all the ``v_X``
vectors at once and therefore must use the form ``dot(R_X,A.T)``.
The shape of ``R_X`` doesn't matter, as long as the last dimension matches the
dimensions of ``A``, for example ``R_X: (N,M,3), A: (3,3), dot(R_X,A.T): (N,M,3))``.

In the following examples, X and Y have the basis vectors still as
*columns* as in the textbook examples above::

1d: ``R_X.shape = (3,)``::

    # R_X == v_X = [x,y,z] 
    R_Y = dot(A, R_X) 
        = dot(R_X,A.T) 
        = dot(R_X, dot(inv(Y), X).T) 
        = linalg.solve(Y, dot(X, R_X))
        = [x', y', z']

    >>> X=rand(3,3); R_X=rand(3); Y=rand(3,3)
    >>> R_Y1=dot(R_X, dot(inv(Y), X).T)
    >>> R_Y2=linalg.solve(Y, dot(X,R_X))
    >>> R_Y1-R_Y2
    array([ 0.,  0.,  0.])

2d: ``R_X.shape = (N,3)``::

    # Array of coords of N atoms, R_X[i,:] = coord of i-th atom. The dot
    # product is broadcast along the first axis of R_X (i.e. *each* row of R_X is
    # dot()'ed with A.T)::

    R_X = [[x0,       y0,     z0],
           [x1,       y1,     z1],
            ...
           [x(N-1),   y(N-1), z(N-1)]]
    R_Y = dot(R,A.T) = 
          [[x0',     y0',     z0'],
           [x1',     y1',     z1'],
            ...
           [x(N-1)', y(N-1)', z(N-1)']]

    >>> X=rand(3,3); R_X=rand(5,3); Y=rand(3,3)
    >>> R_Y1=dot(R_X, dot(inv(Y), X).T) 
    >>> R_Y2=linalg.solve(Y, dot(R_X,X.T).T).T
    >>> R_Y1-R_Y2
    array([[ -3.05311332e-16,   2.22044605e-16,   4.44089210e-16],
           [  4.44089210e-16,   1.11022302e-16,  -1.33226763e-15],
           [ -4.44089210e-16,   0.00000000e+00,   1.77635684e-15],
           [  2.22044605e-16,   2.22044605e-16,   0.00000000e+00],
           [ -2.22044605e-16,   0.00000000e+00,   0.00000000e+00]])

Here we used the fact that ``linalg.solve`` can solve for many rhs's at the
same time (``Ax=b, A:(M,M), b:(M,N)`` where the rhs's are the columns of ``b``).

3d: ``R_X.shape = (nstep, natoms, 3)``::

    # R_X[istep, iatom,:] is the shape (3,) vec of coords for atom `iatom` at
    # time step `istep`.
    # R_X[istep,...] is a (nstep,3) array for this time step. Then we can use
    # the methods for the 2d array above.
    # Broadcasting along the first and second axis. 
    # These loops have the same result as R_Y=dot(R_X, A.T)::
    for istep in xrange(R_X.shape[0]):
        R_Y[istep,...] = dot(R_X[istep,...],A.T)
    
Here, ``linalg.solve`` cannot be used b/c ``R_X`` is 3d and ``A`` has to be
calculated using the inverse, which is mildly unpleasent. 

The above loops are implemented in ``flib.f90`` for the special case fractional
<-> cartesian, using only dot products and ``linalg.solve`` in each loop.

Notes for the special case fractional <-> cartesian
---------------------------------------------------

We have again

.. math::

    Y v_Y = X v_X

    v_Y = Y^{-1}  X v_X = A v_X
    
    v_Y^T = (A v_X)^T = v_X^T . A^T

Now with :math:`X=I`, frac -> cart is merily the dot product

.. math::
    v_X^T = v_Y^T Y^T

and cart -> frac is simply

.. math::
    v_Y^T = v_X^T . (Y^{-1})^T

Note that :math:`(Y^{-1})^T = (X^T)^{-1}`, so if you have :math:`Y` already as rows, then
the transpose can be omitted    

.. math::
    v_Y^T = v_X^T . Y^{-1}
