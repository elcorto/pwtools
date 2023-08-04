#!/usr/bin/env python3

import numpy as np
from pwtools import io, crys, common

lmp_in = {}
lmp_in['md-nvt'] = """
clear
units metal
boundary p p p
atom_style atomic

read_data lmp.struct

### interactions
pair_style tersoff
pair_coeff * * ../AlN.tersoff Al N

### IO
dump dump_txt all custom 1 lmp.out.dump id type xu yu zu fx fy fz &
    vx vy vz xsu ysu zsu
dump_modify dump_txt sort id
dump dump_dcd all dcd 1 lmp.out.dcd
dump_modify dump_dcd sort id unwrap yes
thermo_style custom step temp vol cella cellb cellc cellalpha cellbeta cellgamma &
                    ke pe etotal &
                    press pxx pyy pzz pxy pxz pyz cpu
thermo_modify flush yes
thermo 1

### init
velocity all create 300.0 123 rot yes dist gaussian

# run

fix fix_nvt all nvt temp 3000 3000 0.01 tchain 4 &
    mtk yes scaleyz no scalexz no scalexy no flip no

timestep 0.5e-3
run 100
"""

lmp_in['md-npt'] = """
clear
units metal
boundary p p p
atom_style atomic

read_data lmp.struct

### interactions
pair_style tersoff
pair_coeff * * ../AlN.tersoff Al N

### IO
dump dump_txt all custom 1 lmp.out.dump id type xu yu zu fx fy fz &
    vx vy vz xsu ysu zsu
dump_modify dump_txt sort id
dump dump_xyz all xyz 1 lmp.out.xyz
dump_modify dump_xyz element Al N
dump dump_dcd all dcd 1 lmp.out.dcd
dump_modify dump_dcd sort id unwrap yes
thermo_style custom step temp vol cella cellb cellc cellalpha cellbeta cellgamma &
                    ke pe etotal &
                    press pxx pyy pzz pxy pxz pyz cpu
thermo_modify flush yes
thermo 1

### init
velocity all create 300.0 123 rot yes dist gaussian

# run

fix fix_npt all npt temp 3000 3000 0.01 tri 0 0 0.3 tchain 4 pchain 4 &
    mtk yes scaleyz no scalexz no scalexy no flip no

timestep 0.5e-3
run 100
"""

lmp_in['vc-relax'] = """
clear
units metal
boundary p p p
atom_style atomic

read_data lmp.struct

### interactions
pair_style tersoff
pair_coeff * * ../AlN.tersoff Al N

### IO
dump dump_txt all custom 1 lmp.out.dump id type xu yu zu fx fy fz &
    vx vy vz xsu ysu zsu
dump_modify dump_txt sort id
dump dump_dcd all dcd 1 lmp.out.dcd
dump_modify dump_dcd sort id unwrap yes
thermo_style custom step temp vol cella cellb cellc cellalpha cellbeta cellgamma &
                    ke pe etotal &
                    press pxx pyy pzz pxy pxz pyz cpu press
thermo_modify flush yes
thermo 1

fix 1 all box/relax tri 0.0
minimize 1e-8 1e-8 5000 10000
"""

st = crys.Structure(coords_frac=np.array([[0.0]*3, [.5]*3]),
                    cryst_const=np.array([2.85]*3 + [60]*3),
                    symbols=['Al','N'])

for dr in ['md-nvt', 'md-npt', 'vc-relax']:
    common.system("rm -rfv {dr}; mkdir -v {dr}".format(dr=dr))
io.write_lammps('vc-relax/lmp.struct', st)
io.write_lammps('md-nvt/lmp.struct', crys.scell(st,(2,2,2)))
io.write_lammps('md-npt/lmp.struct', crys.scell(st,(2,2,2)))

for dr,txt in lmp_in.items():
    common.file_write('%s/lmp.in' %dr, txt)
