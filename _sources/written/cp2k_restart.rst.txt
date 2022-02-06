.. _cp2k_restart:

How to do a restart in cp2k
===========================

Some comments
-------------

Suggested naming convention for runs::

    calc/0.0 # equilibration run
    calc/0.1 # production, restart of 0.0
    calc/0.2 # production, restart of 0.1
    ...

After the equi run with a strong thermostat such as::

    motion/md/thermostat/csvr/timecon=0.1

do the production run in NVE or NVT with a weakly coupled thermostat like::

    motion/md/thermostat/csvr/timecon=1000

Restart
-------

Each run ``calc/0.0``, ``calc/0.1`` etc will by default write files like::

    cp2k.in
    cp2k.out
    job.script
    PROJECT-1.cell
    PROJECT-1.ener
    PROJECT-1.restart
    PROJECT-1.stress
    PROJECT-frc-1.xyz
    PROJECT-pos-1.xyz
    PROJECT-vel-1.xyz

To restart for the first time, copy only ``cp2k.in`` and the restart file
``PROJECT-1.restart`` to a new location::

    $ mkdir 0.1
    $ cp 0.0/{cp2k.in,job*,PROJECT-1.restart} 0.1/

Now, add the restart section to the input file. Also change the ensemble (e.g.
``nvt`` -> ``nve``) or thermostat settings if needed::

    $ cd 0.1
    $ vim cp2k.in
        &ext_restart
            restart_file_name PROJECT-1.restart
        &end ext_restart
        ...
        &motion
            &md
                ensemble nvt
                steps 100000
                ...
                &thermostat
                    type csvr
                    &csvr
                        timecon 1000
                    &end csvr
                &end thermostat
                ...
            &end md
        &end motion
    $ qsub job.script

For all subsequent restarts, ``0.2``, ``0.3``, ... you only copy the last
restart file and the input with unchanged settings::

    $ mkdir 0.2
    $ cp 0.1/{cp2k.in,job*,PROJECT-1.restart} 0.2/
    $ cd 0.2
    $ qsub job.script


Analysis of runs with pwtools
-----------------------------

The ``*.xyz`` files and the ``*{cell,ener,stress}`` files are written at each
step. However, for efficiency, the restart file ``PROJECT-1.restart`` is
typically written only every 5, 10, or 50 steps. So, one needs to truncate the
files before parsing by using the script ``cut-cp2k.sh`` from `pwtools
<cut_cp2k_>`_::

    $ for d in calc/0.1 calc/0.2; do cut-cp2k.sh $d; done

Now, one can parse each run ``0.1``, ``0.2``, ... separately::

    $ python
    >>> from pwtools import io,crys
    >>> tr1=io.read_cp2k_md('calc/0.1/cp2k.out')
    >>> tr2=io.read_cp2k_md('calc/0.2/cp2k.out')
    ...

and then concatenate the trajectories::

    >>> tr=crys.concatenate([tr1,tr2])

or more compact::

    >>> tr=crys.concatenate([io.read_cp2k_md('calc/%s/cp2k.out' %x) \
    ...                      for x in ['0.1', '0.2'])

Now, do fun stuff::

    >>> plot(tr.temperature)
    >>> plot(tr.etot)
    >>> plot(tr.etot+ekin)
    >>> plot(tr.pressure)
    >>> plot(tr.coords[...,0], 'b') # all x-coords of all atoms
    >>> d=crys.rpdf(tr, amask=['O','H']); plot(d[:,0], d[:,2])

.. _cp2k_restart_www: http://www.cp2k.org/restarting
.. _cut_cp2k: pwtools/bin/cut-cp2k.sh
