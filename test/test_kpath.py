import os, warnings
import numpy as np
from pwtools import kpath, mpl

def test_kpath():
    # only API
    vecs = np.random.rand(10,3)
    kpath.kpath(vecs, N=15)

def test_plot_dis():
    spp = kpath.SpecialPointsPath(ks=np.array([[0,0,0], [.5,0,0], [.7,0,0]]),
                                  symbols=['A','B', 'C'])
    path_norm = np.linspace(0,1,100)
    nfreq = 5
    freqs = np.random.rand(100, nfreq) 
    try:
        print os.environ['DISPLAY']
        # create fig,ax inside
        fig1,ax1 = kpath.plot_dis(path_norm, freqs, spp)
        # pass ax from outside, returns fig2,ax2 but we don't use that b/c ax
        # is in-place modified
        fig2,ax2 = mpl.fig_ax()
        kpath.plot_dis(path_norm, freqs, spp, ax=ax2)
        lines1 = ax1.get_lines()
        lines2 = ax2.get_lines()
        for idx in range(nfreq):
            x1 = lines1[idx].get_xdata()
            x2 = lines2[idx].get_xdata()
            y1 = lines1[idx].get_ydata()
            y2 = lines2[idx].get_ydata()
            assert (x1 == x2).all()
            assert (y1 == y2).all()
    except KeyError:
        warnings.warn("no DISPLAY environment variable, skipping test")

