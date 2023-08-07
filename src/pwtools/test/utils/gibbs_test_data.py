#!/usr/bin/env python3

import os
import numpy as np
from itertools import product
from socket import gethostname
from pwtools.thermo import Gibbs
from pwtools.mpl import plt
from pwtools.signal import gauss
from pwtools import num, crys, io, common


def filt_dct(dct):
    """Filter None values from dict."""
    return dict((k, v) for k, v in dct.items() if v is not None)


files_dir = os.path.abspath("../../../../test/files/")
pj = os.path.join

if __name__ == "__main__":
    # number of varied axis points
    nax = 6
    # phonon freq axis
    freq = np.linspace(0, 1000, 300)  # cm^-1
    # use linear T axis to avoid numerical noise in derivatives (alpha_V, ...) at low
    # T, which we would see when using a log scale
    T = np.linspace(5, 2000, 50)  # K
    P = np.linspace(0, 5, 2)  # GPa

    # 1d case
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
    common.makedirs(pj(files_dir, "gibbs/1d"))
    io.write_h5(
        pj(files_dir, "gibbs/1d/%s.h5" % gethostname()), filt_dct(g), mode="w"
    )

    # 2d case
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
    common.makedirs(pj(files_dir, "gibbs/2d"))
    io.write_h5(
        pj(files_dir, "gibbs/2d/%s.h5" % gethostname()), filt_dct(g), mode="w"
    )

    # fake 3d case
    case = "1d"
    cell_a = np.linspace(2.5, 3.5, nax)  # Ang
    cell_b = np.linspace(3, 3.8, nax)  # Ang
    cell_c = np.linspace(2, 3, nax)  # Ang
    # e.g. orthorhombic, all angles 90
    volfunc_ax = lambda x: x[0] * x[1] * x[2]
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
    common.makedirs(pj(files_dir, "gibbs/3d-fake-1d"))
    io.write_h5(
        pj(files_dir, "gibbs/3d-fake-1d/%s.h5" % gethostname()),
        filt_dct(g),
        mode="w",
    )
