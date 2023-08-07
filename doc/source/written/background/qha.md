# Quasi-harmonic approximation

See {class}`~pwtools.thermo.HarmonicThermo` +
`test/test_qha.py`, {class}`~pwtools.thermo.Gibbs` + `tesst/test_gibbs.py`.

The distinctive feature of the pwtools implementation is that in
{class}`~pwtools.thermo.Gibbs`, we treat unit cell axes instead of just the
volume, allowing to calculate per-axis thermal expansion rates. See for
instance [Schmerler and Kortus, Phys. Rev. B 89,
064109](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.064109).

When only one unit cell parameter is scaled (e.g. `a` in case of cubic
cells), we call this the "1d" case. If two unit cell axes are scaled
independently on a grid (say `a` and `c` in a hexagonal setting), this
is called "2d" case.

Example results from `Gibbs` for rocksalt AlN (vary cell parameter `a` = `ax0`
of the primitive cell, so 1d case). `Gibbs` returns a dictionary with keys
named as below (such as `/#opt/P/ax0`). The naming scheme is explained in
{class}`~pwtools.thermo.Gibbs`. Below is the result of inspecting an HDF5 file
written by

```py
# See /path/to/pwtools/src/pwtools/test/utils/gibbs_test_data.py
gibbs = Gibbs(...)
g = gibbs.calc_G(calc_all=True)
pwtools.io.write_h5("thermo.h5", g)
```

which we can query for array shapes like so:

```
$ h5ls -rl thermo.h5 | grep Dataset
/#opt/P/B                Dataset {21}
/#opt/P/H                Dataset {21}
/#opt/P/V                Dataset {21}
/#opt/P/ax0              Dataset {21}
/#opt/T/P/B              Dataset {200, 21}
/#opt/T/P/Cp             Dataset {200, 21}
/#opt/T/P/G              Dataset {200, 21}
/#opt/T/P/V              Dataset {200, 21}
/#opt/T/P/alpha_V        Dataset {200, 21}
/#opt/T/P/alpha_ax0      Dataset {200, 21}
/#opt/T/P/ax0            Dataset {200, 21}
/P/P                     Dataset {21}
/P/ax0/H                 Dataset {21, 51}
/T/P/ax0/G               Dataset {200, 21, 51}
/T/T                     Dataset {200}
/ax0/Etot                Dataset {51}
/ax0/T/Cv                Dataset {51, 200}
/ax0/T/Evib              Dataset {51, 200}
/ax0/T/F                 Dataset {51, 200}
/ax0/T/Fvib              Dataset {51, 200}
/ax0/T/Svib              Dataset {51, 200}
/ax0/V                   Dataset {51}
/ax0/ax0                 Dataset {51, 1}
```


The T,P grid has

* 21 P points (`/P/P`)
* 200 T points (`/T/T`)

Further, we have

* 51 volumes (`/ax0/ax0`, `/ax0/V`), `V=volfunc_ax([ax0]`)
* 51 $C_V$ curves (`/ax0/T/Cv`), one per volume, 200 T grid points each
* same for F, Fvib, etc
* Gibbs energy $\mathcal G(V,T,P)$ = $\mathcal G(\texttt{ax0}, T, P)$: `/T/P/ax0/G`
* minimized Gibbs energy $\min_V \mathcal G(V,T,P) = G(T,P)$: `/#opt/T/P/G` on T,P grid
  (200x21)
* 21 $C_P(T)$ curves (200 T points): `/#opt/T/P/Cp`, where for each P,
  $C_P(T) = -T (∂^2 G(T,P) / ∂T^2)_P$
* `/#opt/P/{B,H,V}`: results from $\min_V H = \min_V(E+PV)$ without phonon
  contributions, at $T=0$

2d case (hexagonal wurtzite AlN, vary cell axis `a` = `ax0` and `c` = `ax1`):

```
$ h5ls -rl thermo.h5 | grep Dataset
/#opt/P/H                Dataset {21}
/#opt/P/V                Dataset {21}
/#opt/P/ax0              Dataset {21}
/#opt/P/ax1              Dataset {21}
/#opt/T/P/Cp             Dataset {200, 21}
/#opt/T/P/G              Dataset {200, 21}
/#opt/T/P/V              Dataset {200, 21}
/#opt/T/P/alpha_V        Dataset {200, 21}
/#opt/T/P/alpha_ax0      Dataset {200, 21}
/#opt/T/P/alpha_ax1      Dataset {200, 21}
/#opt/T/P/ax0            Dataset {200, 21}
/#opt/T/P/ax1            Dataset {200, 21}
/P/P                     Dataset {21}
/P/ax0-ax1/H             Dataset {21, 263}
/T/P/ax0-ax1/G           Dataset {200, 21, 263}
/T/T                     Dataset {200}
/ax0-ax1/Etot            Dataset {263}
/ax0-ax1/T/Cv            Dataset {263, 200}
/ax0-ax1/T/Evib          Dataset {263, 200}
/ax0-ax1/T/F             Dataset {263, 200}
/ax0-ax1/T/Fvib          Dataset {263, 200}
/ax0-ax1/T/Svib          Dataset {263, 200}
/ax0-ax1/V               Dataset {263}
/ax0-ax1/ax0-ax1         Dataset {263, 2}
```

Here we see the individual axis expansion rates $\alpha_a$ =
`/#opt/T/P/alpha_ax0` and $\alpha_c$ = `/#opt/T/P/alpha_ax1` in addition to
$\alpha_V$ = `/#opt/T/P/alpha_V`.

Other tools

* <https://phonopy.github.io/phonopy/>
  * 1d only (vary volume)
  * <https://phonopy.github.io/phonopy/qha.html>
  * also calculates phonons using supercell method (finite diffs), uses VASP by
    default, can also use QE
* several projects listed at <https://github.com/topics/quasi-harmonic-approximation>
  * untested
* <https://github.com/gfulian/quantas>
  * untested
