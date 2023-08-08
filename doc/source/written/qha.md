(qha)=
# Quasi-harmonic approximation

See {class}`~pwtools.thermo.HarmonicThermo` +
`test/test_qha.py`, {class}`~pwtools.thermo.Gibbs` + `tesst/test_gibbs.py`.

The distinctive feature of the pwtools implementation is that in
{class}`~pwtools.thermo.Gibbs`, we treat unit cell axes instead of just the
volume, allowing to calculate per-axis thermal expansion rates. See for
instance [Schmerler and Kortus, Phys. Rev. B 89, 064109][paper].

When only one unit cell parameter is scaled (e.g. `a` in case of cubic cells),
we call this the "1d" case. If two unit cell axes are scaled independently on a
2d grid (say `a` and `c` in a hexagonal setting), this is called "2d" case.
Varying `a`, `b` and `c` on a 3d grid, called "3d" case, is partially supported
but has not been used in practice.

The case where you only have a series of volumes (say from variable cell
relaxations at different pressures) is called "fake 1d" case. See below for
more.

## 1d

See ``test_gibbs.py::test_gibbs_1d`` (using generated data).

Example results from `Gibbs` for rocksalt AlN from [the paper above][paper]
(vary cell parameter `a` = `ax0` of the primitive cell, so 1d case). `Gibbs`
returns a dictionary with keys named as below (such as `/#opt/P/ax0`). The
naming scheme is explained in {class}`~pwtools.thermo.Gibbs`. Below is the
result of inspecting an HDF5 file written by

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

* The (T,P) grid `/T/P` is of size (200,21)
  * 200 T points `/T/T`
  * 21 pressure values `/P/P`

This is the grid you define manually via `Gibbs(T=..., P=...)`.

* 51 cell parameter points `/ax0/ax0` and thus 51 volumes (`/ax0/V`,
  `V=volfunc_ax([ax0]`)

Further, we have

* 51 $C_V$ curves (`/ax0/T/Cv`), one per volume, 200 T grid points each
* same for F, Fvib, etc
* Gibbs energy $\mathcal G(V,T,P)$ = $\mathcal G(\texttt{ax0}, T, P)$: `/T/P/ax0/G`
* minimized Gibbs energy $\min_V \mathcal G(V,T,P) = G(T,P)$: `/#opt/T/P/G` on T,P grid
  (200x21)
* 21 $C_P(T)$ curves (200 T points): `/#opt/T/P/Cp`, where for each P,
  $C_P(T) = -T (∂^2 G(T,P) / ∂T^2)_P$
* `/#opt/P/{B,H,V}`: results from $\min_V H = \min_V(E+PV)$ without phonon
  contributions, at $T=0$

## 2d

See ``test_gibbs.py::test_gibbs_2d`` (using generated data).

2d case (hexagonal wurtzite AlN from [the paper above][paper], vary cell axis
`a` = `ax0` and `c` = `ax1`):

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

* The (T,P) grid `/T/P` is of size (200,21)
  * 200 T points `/T/T`
  * 21 pressure values `/P/P`
* The 2d (a,c) grid `/ax0-ax1/ax0-ax1` has 263 points (of successful Phonon
  calculations), represented as an array of shape (263,2).
* At each (T,P) point, the Gibbs energy will be fitted by `fitfunc["2d-G"]` as
  `G(a,c)`, producing `/#opt/T/P/G` of shape (200,21) .. same shape as the
  `/T/P` grid.

Here we see the individual axis expansion rates $\alpha_a$ =
`/#opt/T/P/alpha_ax0` and $\alpha_c$ = `/#opt/T/P/alpha_ax1` in addition to
$\alpha_V$ = `/#opt/T/P/alpha_V`.


## "Fake" 1d a.k.a. usual QHA with varying volume

See ``test_gibbs.py::test_gibbs_3d_fake_1d`` (using generated data). The HDF5
data below is from that test.

```
$ h5ls -rl thermo.h5 | grep Dataset
/#opt/P/B                Dataset {2}
/#opt/P/H                Dataset {2}
/#opt/P/V                Dataset {2}
/#opt/P/ax0              Dataset {2}
/#opt/P/ax1              Dataset {2}
/#opt/P/ax2              Dataset {2}
/#opt/T/P/B              Dataset {50, 2}
/#opt/T/P/Cp             Dataset {50, 2}
/#opt/T/P/G              Dataset {50, 2}
/#opt/T/P/V              Dataset {50, 2}
/#opt/T/P/alpha_V        Dataset {50, 2}
/#opt/T/P/alpha_ax0      Dataset {50, 2}
/#opt/T/P/alpha_ax1      Dataset {50, 2}
/#opt/T/P/alpha_ax2      Dataset {50, 2}
/#opt/T/P/ax0            Dataset {50, 2}
/#opt/T/P/ax1            Dataset {50, 2}
/#opt/T/P/ax2            Dataset {50, 2}
/P/P                     Dataset {2}
/P/ax0-ax1-ax2/H         Dataset {2, 6}
/T/P/ax0-ax1-ax2/G       Dataset {50, 2, 6}
/T/T                     Dataset {50}
/ax0-ax1-ax2/Etot        Dataset {6}
/ax0-ax1-ax2/T/Cv        Dataset {6, 50}
/ax0-ax1-ax2/T/Evib      Dataset {6, 50}
/ax0-ax1-ax2/T/F         Dataset {6, 50}
/ax0-ax1-ax2/T/Fvib      Dataset {6, 50}
/ax0-ax1-ax2/T/Svib      Dataset {6, 50}
/ax0-ax1-ax2/V           Dataset {6}
/ax0-ax1-ax2/ax0-ax1-ax2 Dataset {6, 3}
```

Here we have only 2 pressure values (`/P/P`), 50 T steps `/T/T`, 6 axis points
`/ax0-ax1-ax2/ax0-ax1-ax2` where we change all 3 axes at once in each step,
therefore we also have 6 volumes (`/ax0-ax1-ax2/V`).

How to supply simple 1d volumes:

tl;dr Use a `(N,2)` or `(N,3)` shaped `axes_flat` but set
`case="1d"`.

This is what most other QHA tools do by default. In the general triclinic
case (or more symmetric such as hexagonal), you can always do a variable
cell relaxation (QE: "vc-relax") for several target pressures and generate
structures that way. Then provide `axes_flat` as shape `(N, 2)` or `(N,
3)` data (the latter is the most generic and always works, independently
of the actual symmetry). For instance use {func}`~pwtools.io.read_pw_scf()`,
{func}`~pwtools.io.read_cif()` or {func}`~pwtools.io.read_pickle()` to access
cell parameters. Calculate the volume with {func}`~pwtools.crys.volume_cc()`.

```py
def volfunc_ax(x):
    """General triclinic case.

        axes_flat.shape = (N,3)
        x = axes_flat[i,...]
        x.shape = (3,)
    """
    assert len(x) == 3
    return pwtools.crys.volume_cc([x[0], x[1], x[2], alpha, beta, gamma])

axes_flat = []
for idx in range(n_volumes):
    # written by pwtools.crys.Structure.dump()
    st = pwtools.io.read_pickle(f"/path/to/calc/{idx}/struct.pk")
    axes_flat.append(st.cryst_const[:3])
axes_flat = np.array(axes_flat)

gibbs = Gibbs(..., case="1d")
g = gibbs.calc_G(calc_all=True)

# Cp(T) for all P
plot(g["/T/T"], g["/#opt/T/P/Cp"])
```

Set `Gibbs(..., case="1d")`. This will calculate `V[i] =
volfunc_ax[axes_flat[i,...]]` and do a 1D fit `G(V)`. You still need to
supply `volfunc_ax`, but this is straight forward as shown above for the
most general triclinic case. You have to know the cell angles and that's it
(we always assume that all cell angles are constant during compression and
expansion).


Other ways to access cell parameters:

```py
# Written by pwtools.io.write_cif()
st = pwtools.io.read_cif(f"/path/to/calc/{idx}/struct.cif")

# Lattice params in pw.out files are not very accurate, better
# store generated structures in another file format and read from
# there. But anyway this is handy to access total energy, stress tensor and
# other DFT results.
st = pwtools.io.read_pw_scf(f"/path/to/calc/{idx}/pw.out")

# Anything that ASE can read. See https://wiki.fysik.dtu.dk/ase/ase/io/io.html
st = pwtools.crys.atoms2struct(ase.io.read(...))
```

For generating structures and post-processing calculations beyond simple loops,
see {ref}`param_study` with {mod}`~pwtools.batch` or a more modern version of
that in [psweep](https://github.com/elcorto/psweep).


## Other tools

* <https://phonopy.github.io/phonopy/>
  * 1d only (vary volume)
  * <https://phonopy.github.io/phonopy/qha.html>
  * also calculates phonons using supercell method (finite diffs), uses VASP by
    default, can also use QE
* <https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html#crystals>
  * `ase.thermochemistry.CrystalThermo` is like
    {class}`~pwtools.thermo.HarmonicThermo` but w/o $C_V$
    ({meth}`~pwtools.thermo.HarmonicThermo.isochoric_heat_capacity`)
* several projects listed at <https://github.com/topics/quasi-harmonic-approximation>
  * untested
* <https://github.com/gfulian/quantas>
  * untested

[paper]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.064109 "Schmerler and Kortus, Phys. Rev. B 89, 064109"
