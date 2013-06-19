Parsing of output from computational codes
==========================================

We have a lot of machinery in :mod:`~pwtools.parse` to parse PWscf and CPMD
output files. We try to return the "natural" units of each code (PWscf: Ry,
Bohr, kbar; CPMD: Ha, Bohr, kbar). For PWscf, we also detect things like
"ATOMIC_POSITIONS crystal | alat | bohr" and transform accordingly.
Nevertheless, *always* verify that the units you get are the ones you expect!

You can also use the short-cut methods ``io.read*()``, which will return
crys.Structure or crys.Trajectory with units eV, Angstrom, ...
