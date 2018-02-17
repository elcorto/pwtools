import os
import numpy as np
from pwtools import kpath, mpl
from pwtools.test import tools

def test_kpath():
    # only API
    vecs = np.random.rand(10,3)
    kpath.kpath(vecs, N=15)

def test_plot_dis():
    spp = kpath.SpecialPointsPath(ks=np.array([[0,0,0], [.5,0,0], [.7,0,0]]),
                                  ks_frac=np.array([[0,0,0], [.55,0,0], [.77,0,0]]),  
                                  symbols=['A','B', 'C'])
    path_norm = np.linspace(0,1,100)
    nfreq = 5
    freqs = np.random.rand(100, nfreq) 
    try:
        print(os.environ['DISPLAY'])
        fig1,ax1,axdos1 = kpath.plot_dis(path_norm, freqs, spp,
                                         show_coords='cart')
        assert axdos1 is None
        fig2,ax2 = mpl.fig_ax()
        kpath.plot_dis(path_norm, freqs, spp, ax=ax2, show_coords='frac')
        lines1 = ax1.get_lines()
        lines2 = ax2.get_lines()
        for idx in range(nfreq):
            x1 = lines1[idx].get_xdata()
            x2 = lines2[idx].get_xdata()
            y1 = lines1[idx].get_ydata()
            y2 = lines2[idx].get_ydata()
            assert (x1 == x2).all()
            assert (y1 == y2).all()
        faxis = np.linspace(freqs.min(), freqs.max(), 30)                
        dos = np.array([faxis, np.random.rand(len(faxis))]).T            
        fig3,ax3,ax3dos = kpath.plot_dis(path_norm, freqs, spp, dos=dos,
                                         show_coords=None)
        # plot 90 rotated -> x and y swapped
        assert (ax3dos.get_lines()[0].get_xdata() == dos[:,1]).all()
        assert (ax3dos.get_lines()[0].get_ydata() == dos[:,0]).all()
    except KeyError:
        tools.skip("no DISPLAY environment variable, skipping test")

