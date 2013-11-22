Coordinate transformation
=========================

Textbook relations and basic linalg
-----------------------------------

:math:`X`, :math:`Y` are square matrices with basis vecs as *rows*

| :math:`X` ... old, shape: (3,3) .. or (M,M) in gerenal
| :math:`Y` ... new, shape: (3,3)
| :math:`I` ... identity matrix, basis vecs of cartesian system, shape: (3,3)
| :math:`C` ... transformation matrix, shape(3,3)
| :math:`v_X` ... row vector v in basis X, shape: (1,3)
| :math:`v_Y` ... row vector v in basis Y, shape: (1,3)
| :math:`v_I` ... row vector v in basis I, shape: (1,3)

We have

.. math::
    v_Y Y = v_X X = v_I I = v_I

    v_Y = v_X X Y^{-1} = v_X C

where we call :math:`C = X Y^{-1}` the transformation matrix.

Every product :math:`v_X X; v_Y Y; v_I I` is actually an expansion of
:math:`v_{X,Y,...}` in the basis vectors contained in :math:`X,Y,...` . If the
dot product is computed, we always get :math:`v` in cartesian coords. 

We now switch to code examples, where :math:`v_X` == ``v_X``.

In general, we don't have one vector ``v_X`` but an array ``R_X`` of shape
(N,M) of row vectors::
    
    R_X = [[--- v_X0 ---],
           [--- v_X1 ---],
           ...
           [-- v_XN-1 --]]

We want to use fast numpy array broadcasting to transform all the ``v_X``
vectors at once.
The shape of ``R_X`` doesn't matter, as long as the last dimension matches the
dimensions of ``C``, for example ``R_X: (N,M,3), C: (3,3), dot(R_X,C): (N,M,3))``.

Examples:

1d: ``R_X.shape = (3,)``::

    >>> # R_X == v_X = [x,y,z] 
    >>> # R_Y == v_Y = [x',y',z'] 
    >>> X=rand(3,3); R_X=rand(3); Y=rand(3,3); C=dot(X, inv(Y))
    >>> R_Y1=dot(R_X, C)
    >>> R_Y2=dot(dot(R_X, X), inv(Y))
    >>> R_Y3=linalg.solve(Y.T, dot(R_X, X).T)
    >>> print R_Y1-R_Y2
    >>> print R_Y1-R_Y3
    array([ 0.,  0.,  0.])
    array([ 0.,  0.,  0.])

2d: ``R_X.shape = (N,3)``::

    >>> # Array of coords of N atoms, R_X[i,:] = coord of i-th atom. The dot
    >>> # product is broadcast along the first axis of R_X (i.e. *each* row of R_X is
    >>> # dot()'ed with C)::
    >>> #   
    >>> # R_X = [[x0,       y0,     z0],
    >>> #        [x1,       y1,     z1],
    >>> #         ...
    >>> #        [x(N-1),   y(N-1), z(N-1)]]
    >>> # R_Y = [[x0',     y0',     z0'],
    >>> #        [x1',     y1',     z1'],
    >>> #         ...
    >>> #        [x(N-1)', y(N-1)', z(N-1)']]
    >>> #
    >>> X=rand(3,3); R_X=rand(5,3); Y=rand(3,3); C=dot(X, inv(Y))
    >>> R_Y1=dot(R_X, C)
    >>> R_Y2=dot(dot(R_X, X), inv(Y))
    >>> R_Y3=linalg.solve(Y.T, dot(R_X, X).T).T
    >>> assert np.allclose(R_Y1-R_Y2, R_Y1-R_Y3)
    >>> print R_Y1-R_Y3
    [[  1.33226763e-15  -8.88178420e-16  -2.77555756e-16]
     [  2.22044605e-15  -3.41393580e-15  -3.88578059e-16]
     [  1.77635684e-15  -8.88178420e-16  -3.33066907e-16]
     [  6.66133815e-16  -2.22044605e-16  -7.58941521e-17]
     [  6.66133815e-16  -5.55111512e-16   0.00000000e+00]]
    

Here we used the fact that ``linalg.solve`` can solve for many rhs's at the
same time (``Ax=b, A:(M,M), b:(M,N)`` where the rhs's are the columns of
``b``). The result from ``linalg.solve`` has the same shape as ``b``:
``(M,N)``, i.e. each result vector is a column. That's why we need the last
transpose.

3d: ``R_X.shape = (nstep, natoms, 3)``::

    >>> # R_X[istep, iatom,:] is the shape (3,) vec of coords for atom `iatom` at
    >>> # time step `istep`.
    >>> # R_X[istep,...] is a (nstep,3) array for this time step. Then we can use
    >>> # the methods for the 2d array above.
    >>> for istep in xrange(R_X.shape[0]):
    >>>     R_Y[istep,...] = dot(dot(R_X[istep,...], X), inv(Y))
    
Here, ``linalg.solve`` cannot be used b/c ``R_X`` is 3d and ``C`` has to be
calculated using the inverse, which is mildly unpleasent. 

The above loops are implemented in ``flib.f90`` for the special case fractional
<-> cartesian, using only dot products and linear system solvers from Lapack 
in each loop.

Notes for the special case fractional <-> cartesian
---------------------------------------------------

We have again

.. math::
    v_Y Y = v_X X

    v_Y = v_X X Y^{-1} = v_X C

With :math:`X=I`, we define :math:`v_X` cartesian and :math:`v_Y` fractional
coordinates. Then the transform fractional -> cartesian is the dot product

.. math::
    v_Y Y = v_X

as already stated above and cartesian -> fractional is

.. math::
    v_Y = v_X Y^{-1}

which is the solution of the linear system :math:`v_Y Y = v_X`. It cannot get
more simple.

Row vs. column form
-------------------
We use a row oriented form of all relations, where :math:`v_X` are row vectors
(1,M) and :math:`X` has basis vectors as rows. This is optimal for direct
translation to numpy code, where we can use broadcasting.

In the math literature you find the column oriented form, where :math:`X`
has the basis vectors as columns and :math:`v_X` is a column vector
(M,1). Then we write :math:`X v_X = Y v_Y` and all formulas here translate by
:math:`(A x)^T = x^T A^T` and :math:`(A^{-1})^T = (A^T)^{-1}`.

Tools like ``linalg.solve`` and Lapack solvers assume the column oriented form
:math:`A x = b` rather than our row oriented form :math:`x A = b`, so you need
to use ``linalg.solve(A.T, b.T)``.

