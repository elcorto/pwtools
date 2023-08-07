"""
Compare thermo.Gibbs results with pre-calculated reference data. Need this in
case of code refactoring.

In the 2d case, we need to use fairly high tolerances (rtol). We have numerical
noise in the test/files/gibbs/**.h5 test data files when running
test/utils/gibbs_test_data.py on different machines, with the same numpy, scipy
etc versions! But only in the 2d case. All #opt results are affected. This is a
comparison of

  files/gibbs/2d/cartman.h5
  files/gibbs/2d/kenny.h5

  value                 abs. err   rel. err
  -----                 --------   --------
  /#opt/T/P/Cp          1.04e-10   4.18e-11
  /#opt/T/P/G              5e-16   5.36e-16  <<<
  /#opt/T/P/V           3.02e-07   9.78e-09
  /#opt/T/P/alpha_V     7.08e-10   0.000187  <<<
  /#opt/T/P/alpha_ax0   3.26e-10   0.000206  <<<
  /#opt/T/P/alpha_ax1   2.21e-10   0.000356  <<<
  /#opt/T/P/ax0         1.43e-08   4.74e-09
  /#opt/T/P/ax1         6.76e-09   1.98e-09
  /P/P                         0          0
  /T/P/ax0-ax1/G               0          0
  /T/T                         0          0
  /ax0-ax1/Etot                0          0
  /ax0-ax1/T/Cv                0          0
  /ax0-ax1/T/Evib              0          0
  /ax0-ax1/T/F                 0          0
  /ax0-ax1/T/Fvib              0          0
  /ax0-ax1/T/Svib              0          0
  /ax0-ax1/V                   0          0
  /ax0-ax1/ax0-ax1             0          0

Since all #opt results are affected, it seems that fmin(), which is used by
default in the 2d case to minimize G(ax0,...), is very sensitive to rounding
errors and therefore the results depend on the machine. But note that
/#opt/T/P/ax0 and ax1 have an error of about 1e-9, which is still OK. The only
big relative error of 1e-4 is in the alpha_* values! This is probably b/c the
errors in the ax0 and ax1 results are magnified by the 1st derivative. Similar
for Cp, where the error in G is 1e-16 and in Cp (*second* derivative) is 1e-11,
only that this is still pretty good!

For this reason, it is also a bad idea to use a log-scale T axis in this test.
At low T, the thermal expansion is zero and almost constant. Since we use a
Spline by default to fit the ax0(T) and ax1(T), which goes thru all points,
small numerical noise *on one machine* from the minimization (find
/#opt/T/P/ax0 and ax1) will be greatly magnified when taking the derivative,
especially if we have very small dT steps. This would be no problem if the
produced data was the same on each machine! Then we would conmpare the *same*
large fluctuations. However, due to the rounding behavior on different
machines, these (large) fluctuations are *different* for each machine, which
makes the resulting alpha values impossible to compare at low T. When using a
linear T scale, we kind of average out the fluctuations and therefore minimize
this effect.

In the 1d case, we have *zero* difference, i.e. different machines produce the
same test data.
"""

import os
from itertools import product

import numpy as np
import pytest

from pwtools.thermo import Gibbs
from pwtools.signal import gauss
from pwtools import num, io, constants
from pwtools.test import tools


def compare_dicts_with_arrays(a, b):
    """For each numpy array in `a` and `b`, print absolute and relative
    difference."""
    for k in list(a.keys()):
        if k in b and b[k] is not None:
            df = abs(a[k] - b[k]).max()
            fmt = "{:20}  {:8.3g}   {:8.3g}"
            print(fmt.format(k, df, df / abs(a[k]).max()))


def test_gibbs_1d():
    # number of varied axis points
    nax = 6
    # phonon freq axis
    freq = np.linspace(0, 1000, 300)  # cm^-1
    T = np.linspace(5, 2000, 50)  # K
    P = np.linspace(0, 5, 2)  # GPa

    case = "1d"
    V = np.linspace(10, 20, nax)
    axes_flat = V ** (1 / 3.0)  # cubic
    volfunc_ax = lambda x: x[0] ** 3.0
    etot = (V - V.mean()) ** 2
    fcenter = 450 + 100 * (axes_flat - axes_flat.min())
    # fake phonon dos data (Gaussian), shift to lower freq for higher volume
    phdos = [np.array([freq, gauss(freq - fc, 100)]).T for fc in fcenter[::-1]]

    gibbs = Gibbs(
        T=T,
        P=P,
        etot=etot,
        phdos=phdos,
        axes_flat=axes_flat,
        volfunc_ax=volfunc_ax,
        case=case,
        dosarea=None,
    )
    gibbs.set_fitfunc(
        "C", lambda x, y: num.Spline(x, y, s=None, k=5, eps=1e-5)
    )
    g = gibbs.calc_G(calc_all=True)

    dr = "files/gibbs/1d"
    for name in os.listdir(dr):
        fn = "%s/%s" % (dr, name)
        gref = io.read_h5(fn)
        print("testing: %s" % fn)
        compare_dicts_with_arrays(gref, g)
        tools.assert_dict_with_all_types_almost_equal(
            gref, g, keys=list(gref.keys()), atol=1e-14, rtol=1e-8
        )

    # test enthalpy stuff for 1d case
    # E(V)
    ev = num.PolyFit1D(g["/ax0/V"], g["/ax0/Etot"], deg=5)
    # P(V)
    pv = lambda v: -ev(v, der=1) * constants.eV_by_Ang3_to_GPa
    assert np.allclose(g["/P/P"], pv(g["/#opt/P/V"]))
    assert np.allclose(
        g["/#opt/P/H"],
        ev(g["/#opt/P/V"])
        + g["/P/P"] * g["/#opt/P/V"] / constants.eV_by_Ang3_to_GPa,
    )


def test_gibbs_2d():
    # number of varied axis points
    nax = 6
    # phonon freq axis
    freq = np.linspace(0, 1000, 300)  # cm^-1
    T = np.linspace(5, 2000, 50)  # K
    P = np.linspace(0, 5, 2)  # GPa

    case = "2d"
    cell_a = np.linspace(2.5, 3.5, nax)  # Ang
    cell_c = np.linspace(3, 3.8, nax)  # Ang
    volfunc_ax = lambda x: x[0] ** 2 * x[1]
    axes_flat = np.array([x for x in product(cell_a, cell_c)])
    V = np.array([volfunc_ax(x) for x in axes_flat])  # Ang**3
    cell_a_mean = cell_a.mean()
    cell_c_mean = cell_c.mean()
    etot = np.array(
        [
            (a - cell_a_mean) ** 2.0 + (c - cell_c_mean) ** 2.0
            for a, c in axes_flat
        ]
    )
    phdos = []
    Vmax = V.max()
    for ii in range(axes_flat.shape[0]):
        a, c = axes_flat[ii, :]
        fc = 550 - 50 * V[ii] / Vmax
        phdos.append(np.array([freq, gauss(freq - fc, 100) * 0.01]).T)

    gibbs = Gibbs(
        T=T,
        P=P,
        etot=etot,
        phdos=phdos,
        axes_flat=axes_flat,
        volfunc_ax=volfunc_ax,
        case=case,
        dosarea=None,
    )
    gibbs.set_fitfunc(
        "C", lambda x, y: num.Spline(x, y, s=None, k=5, eps=1e-5)
    )
    g = gibbs.calc_G(calc_all=True)

    dr = "files/gibbs/2d"
    for name in os.listdir(dr):
        fn = "%s/%s" % (dr, name)
        gref = io.read_h5(fn)
        print("testing: %s" % fn)
        compare_dicts_with_arrays(gref, g)
        tools.assert_dict_with_all_types_almost_equal(
            gref, g, keys=list(gref.keys()), atol=1e-8, rtol=1e-3
        )


@pytest.mark.parametrize(
    "case",
    [
        "1d",
        pytest.param("2d", marks=pytest.mark.xfail),
        pytest.param("3d", marks=pytest.mark.xfail),
    ],
)
def test_gibbs_3d_fake_1d(case):
    def volfunc_ax(x):
        assert len(x) == 3
        # e.g. orthorhombic, all angles 90
        return np.prod(x)

    # number of varied axis points
    nax = 6
    # phonon freq axis
    freq = np.linspace(0, 1000, 300)  # cm^-1
    T = np.linspace(5, 2000, 50)  # K
    P = np.linspace(0, 5, 2)  # GPa
    cell_a = np.linspace(2.5, 3.5, nax)  # Ang
    cell_b = np.linspace(3, 3.8, nax)  # Ang
    cell_c = np.linspace(2, 3, nax)  # Ang
    # in 1d case, scale all 3 axes together
    axes_flat = np.array([x for x in zip(cell_a, cell_b, cell_c)])
    V = np.array([volfunc_ax(x) for x in axes_flat])  # Ang**3
    cell_a_mean = cell_a.mean()
    cell_b_mean = cell_b.mean()
    cell_c_mean = cell_c.mean()
    etot = np.array(
        [
            (a - cell_a_mean) ** 2.0
            + (a - cell_b_mean) ** 2.0
            + (c - cell_c_mean) ** 2.0
            for a, b, c in axes_flat
        ]
    )
    phdos = []
    Vmax = V.max()
    for ii in range(axes_flat.shape[0]):
        fc = 550 - 50 * V[ii] / Vmax
        phdos.append(np.array([freq, gauss(freq - fc, 100) * 0.01]).T)

    gibbs = Gibbs(
        T=T,
        P=P,
        etot=etot,
        phdos=phdos,
        axes_flat=axes_flat,
        volfunc_ax=volfunc_ax,
        case=case,
        dosarea=None,
    )
    gibbs.set_fitfunc(
        "C", lambda x, y: num.Spline(x, y, s=None, k=5, eps=1e-5)
    )
    g = gibbs.calc_G(calc_all=True)

    dr = "files/gibbs/3d-fake-1d"
    for name in os.listdir(dr):
        fn = "%s/%s" % (dr, name)
        gref = io.read_h5(fn)
        print("testing: %s" % fn)
        compare_dicts_with_arrays(gref, g)
        tools.assert_dict_with_all_types_almost_equal(
            gref, g, keys=list(gref.keys()), atol=1e-8, rtol=1e-3
        )
