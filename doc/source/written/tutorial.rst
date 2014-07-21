Tutorial
========

Here we collect random short code snippets showing how to perform various
common tasks.

Using ASE?
----------
No problem, use :func:`~pwtools.crys.atoms2struct` and
:func:`~pwtools.crys.struct2atoms` to convert back and forth. If you have a
:class:`~pwtools.crys.Structure`, you can also use the
:meth:`~pwtools.crys.Structure.get_ase_atoms` method, which is the same as
``struct2atoms(struct)``.

For basic ASE compatibility, you may get away with
:meth:`~pwtools.crys.Structure.get_fake_ase_atoms`. That creates an object
which behaves like ``ase.Atoms`` without the need to have ASE installed. 
This is used in :mod:`pwtools.symmetry`, for example.

Find Monkhorst-Pack k-grid sampling for a given unit cell
---------------------------------------------------------

Say you know from a previous convergence study that a k-grid spacing of
``h=0.5`` 1/Angstrom is OK. Now you have a slab or other super cell of your
structure and you want to know "what k-grid do I need to get the same
accuracy". Simple::

    >>> from pwtools import crys
    >>> # new cell in Angstrom
    >>> cell=np.diag([8,8,5])
    >>> crys.kgrid(cell, h=0.5)
    array([2, 2, 3])

OK, so use a :math:`2\times2\times3` MP grid. Instead of defining ``cell`` by
hand, you could also build your structure, have it in a Structure object, say
``st`` and use ``st.cell`` instead.

Parse MD code output, plot stuff
--------------------------------
Lets take cp2k as an example (assuming an interactive Ipython session)::
    
    >>> from pwtools import io
    >>> tr = io.read_cp2k_md('cp2k.out')
    >>> plot(tr.etot)
    >>> figure()
    >>> # x-coord of all atoms over time
    >>> plot(tr.coords[...,0])

.. _avoid_auto_calc:
Avoid auto-calculation for big MD data
--------------------------------------
If you have really big MD data (say several GB), then the :ref:`auto-calculation of
missing properties <container_classes>` might take long and/or fill
up all memory. To avoid that, call the parser explicitly and say
``auto_calc=False`` when creating the :class:`~pwtools.crys.Trajectory`,
which will deactivate auto-calculation. It will only do unit conversion to eV,
Ang, etc. [you can of course also access the parser's attributes directly, e.g.
``pp.coords`` in the unit of the MD code (e.g. Bohr) instead of ``tr.coords``
in Ang].

This is an example for parsing LAMMPS dcd binary data (``log.lammps`` is the
logfile and the default binary file is ``lmp.out.dcd``).

    >>> pp = parse.LammpsDcdMDOutputFile('log.lammps')
    >>> tr = pp.get_traj(auto_calc=False) # default is auto_calc=True

In order to maximally reduce data, you can tell the parser to parse only
certain things::
    
    >>> pp.set_attr_lst(['etot', 'coords', 'temperature'])
    >>> tr = pp.get_traj(auto_calc=False)

You may also use ``auto_calc=True`` here any see what will be
auto-calculated from this minimal input data.

Of course you need to know what can be found in the MD data (e.g. if the MD
code writes no fractional coords, then parsing ``coords_frac`` won't work).

To find out what can be parsed, also check which ``get_*()`` methods the parser
implements (mind also base classes, best is to use Tab completion in ipython:
``>>> pp.get_<tab>`` or have a look at the API documentation).

Binary IO
---------
You can save a :class:`~pwtools.crys.Structure` or
:class:`~pwtools.crys.Trajectory` object as binary file::
    
    >>> # save to binary pickle file
    >>> tr.dump('traj.pk')

and read it back in later using :func:`~pwtools.io.cpickle_load` ::
    
    >>> tr = io.cpickle_load('traj.pk')

which is usually very fast.

Find spacegroup
---------------
Say you have a Trajectory ``tr``, which is the result of a relax calculation and you
want to know the space group of the final optimized structure, namely
``tr[-1]``::

    >>> from pwtools import symmetry
    >>> symmetry.get_spglib_spacegroup(tr[-1], symprec=1e-2)

Easy, eh?

Smoothing a signal or a Trajectory
----------------------------------
Smoothing a signal (usually called "time series") by convolution with another
function and with edge effects handling: :func:`pwtools.signal.smooth`. The same 
can be applied to a Trajectory, which is just a "time series" of Structures.
See :func:`pwtools.crys.smooth`::
    
    >>> a = rand(10000)
    >>> a_smooth = signal.smooth(a, scipy.signal.hann(151))
    >>> tr = Trajectory(...)
    >>> tr_smooth = crys.smooth(tr, scipy.signal.hann(151))

More stuff
----------
* :ref:`dispersion_example`
