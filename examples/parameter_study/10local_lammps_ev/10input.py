#!/usr/bin/env python

# Write lammps input files for a E(V) curve calculation.

import os
import numpy as np
from pwtools import common, batch, sql, crys, lammps

local = batch.Machine(hostname='local',
                      subcmd='bash',
                      scratch='/tmp',
                      filename='calc.templ/job.local',
                      home='/home/schmerler')

templates = [batch.FileTemplate(basename=x) for x in
             ['lmp.in', 'lmp.struct', 'lmp.struct.symbols']]

# rs-AlN
st = crys.Structure(coords_frac=np.array([[0.0]*3, [0.5]*3]),
                    symbols=['Al','N'],
                    cryst_const=np.array([2.78]*3 + [60]*3))

params_lst = []
for target_press in np.linspace(-20,20,15): # GPa, bar in lammps
    params_lst.append([sql.SQLEntry(key='target_press', sqlval=target_press*1e4),
                       sql.SQLEntry(key='struct', sqlval=lammps.struct_str(st)),
                       sql.SQLEntry(key='symbols', sqlval='\n'.join(st.symbols)),
                      ])

calc = batch.ParameterStudy(machines=local,
                            templates=templates,
                            params_lst=params_lst,
                            study_name='lammps_ev',
                            )
calc.write_input(sleep=0, backup=False, mode='w')

if not os.path.exists('calc'):
    os.symlink('calc_local', 'calc')

common.system("cp -r potentials calc_local/")
