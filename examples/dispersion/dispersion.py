#!/usr/bin/env python3

"""
Calculate phonon dispersion with QE's matdyn and plot: example from wz-AlN.

Define special points path in the BZ, construct fine k-path for dispersion,
write matdyn.x input file, call matdyn.x to calculate the dispersion using
the 'q2r.fc' force constants in this dir, load and plot dispersion. Python
rocks!

Note that for this script to work, we need a working QE install (matdyn.x)
and these files:
    q2r.fc
        force constant file from phonon calculation
    pw.out
        SCF output to read atom types and cell to calculate rcell_reduced
"""

import numpy as np
from pwtools import kpath, common, pwscf, io, crys, mpl

sp_symbols = ['$\Gamma$',
              'M',
              'K',
              '$\Gamma$',
              'A',
              ]
# fractional k-space coords for special points
sp_points_frac = np.array([\
    [0,0,0],
    [.5,0,0],
    [1/3., 1/3., 0],
    [0,0,0],
    [0,0,.5],
    ])

# recip. cell in 2*pi/alat units, need this to make QE happy, q-points in
# matdyn.in and matdyn.freq output file are cartesian in 2*pi/alat
st = io.read_pw_scf('pw.out')
rcell_reduced = crys.recip_cell(st.cell) / 2.0 / np.pi * st.cryst_const[0]
sp_points = np.dot(sp_points_frac, rcell_reduced)

# fine path: use N=500 for nice LO-TO split jumps [see below for more comments
# on that]
ks_path = kpath.kpath(sp_points, N=50)

# call matdyn.x
templ_txt = """
&input
    asr='crystal',
XXXMASS
    flfrc='q2r.fc',
    flfrq='XXXFNFREQ'
/
XXXNKS
XXXKS
"""
matdyn_in_fn = 'matdyn.disp.in'
matdyn_freq_fn = 'matdyn.freq.disp'
mass_str = '\n'.join("amass(%i)=%e" %(ii+1,m) for ii,m in \
                      enumerate(st.mass_unique))
rules = {'XXXNKS': ks_path.shape[0],
         'XXXKS': common.str_arr(ks_path),
         'XXXMASS': mass_str,
         'XXXFNFREQ': matdyn_freq_fn,
         }
txt = common.template_replace(templ_txt,
                              rules,
                              conv=True,
                              mode='txt')
common.file_write(matdyn_in_fn, txt)
common.system("gunzip q2r.fc.gz; matdyn.x < %s; gzip q2r.fc" %matdyn_in_fn)

# parse matdyn output and plot

# define special points path, used in plot_dis() to plot lines at special
# points and make x-labels
sp = kpath.SpecialPointsPath(ks=sp_points, ks_frac=sp_points_frac,
                             symbols=sp_symbols)

# QE 4.x, 5.x
ks, freqs = pwscf.read_matdyn_freq(matdyn_freq_fn)
fig,ax,axdos = kpath.plot_dis(kpath.get_path_norm(ks_path), freqs, sp, marker='', ls='-', color='k')

# QE 5.x
##d = np.loadtxt(matdyn_freq_fn + '.gp')
##fig,ax,axdos = kpath.plot_dis(d[:,0], d[:,1:], sp, marker='', ls='-', color='k')

# if needed
#ax.set_ylim(...)

mpl.plt.show()

# Band jumps at Gamma
# -------------------
# Either use many points in kpath() or split sp_points in two sets: sp_points_1
# = something--Gamma and sp_points_2=Gamma--something_else. Do the whole
# call-matdyn-and-parse stuff twice. Parse matdyn.freq.disp for both sets and
# plot both dispersions (for sp_points_1 and sp_points_2) into one plot.
