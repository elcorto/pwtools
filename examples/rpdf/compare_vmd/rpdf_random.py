#!/usr/bin/env python3

import numpy as np
from pwtools import crys, mpl
rand = np.random.rand

# Cubic box with random points and L=20, so rmax_auto=10. We randomly choose
# some atoms to be O, the rest H, which lets us test 2 selections.
#
# For rpdf(), g(r) goes to zero for r > 10 b/c the minimum image convention is
# violated. VMD is correct up to 2*sqrt(0.5)*rmax_auto b/c they apply their
# "spherical cap"-correction.
#
# norm_vmd: For debugging, we also calculate with norm_vmd=True, which results
# in slightly wrong g(r) for the all-all case, while num_int is always correct.
#
# The blue curves rpdf(..., norm_vmd=False) are correct up to rmax_auto.

t1=crys.Trajectory(coords_frac=rand(100,20,3),
                   cell=np.identity(3)*20,
                   symbols=['O']*5+['H']*15)
sy = np.array(t1.symbols)
dr = 0.1
rmax = 25

dct = {'amask': [[sy=='O', sy=='H'], None],
       'sel': [['name O', 'name H'], ['all', 'all']]}

plots = []
for ii in range(2):
    amask = dct['amask'][ii]
    sel = dct['sel'][ii]
    title = sel[0] + ',' + sel[1]

    aa = crys.rpdf(t1, dr=dr, rmax=rmax, amask=amask, norm_vmd=False)
    bb = crys.rpdf(t1, dr=dr, rmax=rmax, amask=amask, norm_vmd=True)
    cc = crys.vmd_measure_gofr(t1, dr=dr, rmax=rmax, sel=sel)

    plots.append(mpl.Plot())
    plots[-1].ax.plot(aa[:,0], aa[:,1], 'b', label="g(r), norm_vmd=False")
    plots[-1].ax.plot(bb[:,0], bb[:,1], 'r', label="g(r), norm_vmd=True")
    plots[-1].ax.plot(cc[:,0], cc[:,1], 'g', label="g(r), vmd")
    plots[-1].legend()
    plots[-1].ax.set_title(title)

    plots.append(mpl.Plot())
    plots[-1].ax.plot(aa[:,0], aa[:,2], 'b', label="int, norm_vmd=False")
    plots[-1].ax.plot(bb[:,0], bb[:,2], 'r', label="int, norm_vmd=True")
    plots[-1].ax.plot(cc[:,0], cc[:,2], 'g', label="int, vmd")
    plots[-1].legend(loc='lower right')
    plots[-1].ax.set_title(title)

mpl.plt.show()
