#!/usr/bin/env python

# Write PWscf input files for a convergence study: vary ecutwfc.

import os
import numpy as np
from pwtools import common, batch, sql, crys, pwscf

theo = batch.Machine(hostname='theo',
                     subcmd='qsub',
                     scratch='/scratch/schmerler',
                     filename='calc.templ/job.pbs.theo',
                     home='/home/schmerler')

templates = [batch.FileTemplate(basename='pw.in')]

# rs-AlN
st = crys.Structure(coords_frac=np.array([[0.0]*3, [0.5]*3]),
                    symbols=['Al','N'],
                    cryst_const=np.array([2.76]*3 + [60]*3))

params_lst = []
for ecutwfc in np.linspace(30,100,8):
    params_lst.append([sql.SQLEntry(key='ecutwfc', sqlval=ecutwfc),
                       sql.SQLEntry(key='ecutrho', sqlval=4.0*ecutwfc),
                       sql.SQLEntry(key='cell', sqlval=common.str_arr(st.cell)),
                       sql.SQLEntry(key='natoms', sqlval=st.natoms),
                       sql.SQLEntry(key='atpos',
                                    sqlval=pwscf.atpos_str(st.symbols,
                                                           st.coords_frac)),
                      ])

calc = batch.ParameterStudy(machines=theo,
                            templates=templates,
                            params_lst=params_lst, 
                            study_name='convergence_test_cutoff',
                            )
calc.write_input(sleep=0, backup=False, mode='w')

if not os.path.exists('calc'):
    os.symlink('calc_theo', 'calc')

common.system("cp -r ../../../test/files/qe_pseudos calc_theo/pseudo; gunzip calc_theo/pseudo/*")
