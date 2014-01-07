#!/usr/bin/env python

import sys
import numpy as np
from pwtools import io, mpl, common, crys
 
npt_txt = """
fix fix_npt all npt temp 3000 3000 0.01 tri 0 0 0.3 tchain 4 pchain 4 &
    mtk yes scaleyz no scalexz no scalexy no flip no
"""

nvt_txt = """
fix fix_nvt all nvt temp 3000 3000 0.01 tchain 4 &
    mtk yes scaleyz no scalexz no scalexy no flip no
"""

lmp_in_templ = """
clear
units metal 
boundary p p p 
atom_style atomic

read_data lmp.struct

### interactions 
pair_style tersoff 
pair_coeff * * AlN.tersoff Al N

### IO
dump dump_txt all custom 1 lmp.out.dump id type xu yu zu fx fy fz &
    vx vy vz xsu ysu zsu 
##dump dump_xyz all xyz 1 lmp.out.xyz
##dump_modify dump_xyz element Al N 
dump dump_dcd all dcd 1 lmp.out.dcd
dump_modify dump_txt sort id 
dump_modify dump_dcd sort id unwrap yes
thermo_style custom step temp vol cella cellb cellc cellalpha cellbeta cellgamma &
                    ke pe etotal &
                    press pxx pyy pzz pxy pxz pyz cpu
thermo_modify flush yes
thermo 1

### init
velocity all create 300.0 123 rot yes dist gaussian

# run
{ensemble}
timestep 0.5e-3
run 1000
"""

assert len(sys.argv) == 2, "need one input arg: nvt or npt"
if sys.argv[1] == 'npt':
    ens_txt = npt_txt
elif sys.argv[1] == 'nvt':    
    ens_txt = nvt_txt
else:
    raise StandardError("only nvt / npt allowed")

# create structure file
st = crys.Structure(coords_frac=np.array([[0.0]*3, [.5]*3]),
                    cryst_const=np.array([2.85]*3 + [60]*3),
                    symbols=['Al','N'])
io.write_lammps('lmp.struct', crys.scell(st,(3,3,3)))

# write lmp.in for nvt or npt
common.file_write('lmp.in', lmp_in_templ.format(ensemble=ens_txt))

# run lammps
common.system("mpirun -np 2 lammps < lmp.in", wait=True)

# read trajectory
trtxt_orig = io.read_lammps_md_txt('log.lammps')
trdcd = io.read_lammps_md_dcd('log.lammps')

# plotting
plots = mpl.prepare_plots(['coords', 'coords_frac', 'velocity', 
                           'cryst_const', 'cell'])
for name,pl in plots.iteritems():
    trtxt = trtxt_orig.copy()
    print name
    xtxt = getattr(trtxt, name)
    setattr(trtxt, name, None)
    xcalc = eval('trtxt.get_%s()' %name)
    if name == 'cell':
        sl = np.s_[Ellipsis]
        func = lambda x: np.reshape(x, (x.shape[0], 9))
    elif name in trtxt.attrs_nstep_3d:
        # coords_frac and coords: only x-coord (index=0)
        sl = np.s_[Ellipsis,0]
        func = lambda x: x
    else:
        sl = np.s_[Ellipsis]
        func = lambda x: x
    lt = pl.ax.plot(func(xtxt[sl]),'b')
    lc = pl.ax.plot(func(xcalc[sl]),'r')
    ld = pl.ax.plot(func(getattr(trdcd, name)[sl]),'g')
    pl.ax.set_title(name)
    pl.ax.legend((lt[0],lc[0],ld[0]), ('txt', 'calc', 'dcd'))

mpl.plt.show()

