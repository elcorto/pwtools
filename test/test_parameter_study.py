# Minimal parameter study example. 
#
# Here, we consider the special case of a commonly needed convergence study of
# cut-off (ecutwfc) and kpoints. It is known that one can vary ecutwfc and
# kpoints *independently* of each other, i.e. instead of a grid:
#
#   25 50 75
# 2 *  *  *
# 4 *  *  *
# 6 *  *  *
#
# just one row and column to save computer time and avoid redundant
# information.
#
#   25 50 75
# 2 *  *  *
# 4 *  
# 6 *  
#
# This is done by the nested_loops() trick below.


import os
import numpy as np
from pwtools import comb, batch, common
from pwtools.pwscf import kpointstr_pwin

from testenv import testdir

def test():    
    pj = os.path.join

    prefix = 'convergence'
    machines = [batch.local]
    templ_dir = 'files/calc.templ'
    templates = \
        {'pw': batch.FileTemplate(basename='pw.in', 
                                  templ_dir=templ_dir), 
        }
    for m in machines:
        templates[m.jobfn] = batch.FileTemplate(basename=m.jobfn,
                                                templ_dir=templ_dir)
    _kpoints = [2,4,6]
    _ecutwfc = np.array([25, 50, 75])
    kpoints = batch.sql_column('kpoints', 'text', [kpointstr_pwin([x]*3) \
                                                   for x in _kpoints])
    ecutwfc = batch.sql_column('ecutwfc', 'float', _ecutwfc)
    ecutrho = batch.sql_column('ecutrho', 'float', 10*_ecutwfc)
    params_lst = comb.nested_loops([[kpoints[0]], zip(ecutwfc, ecutrho)]) + \
                 comb.nested_loops([kpoints[1:], zip([ecutwfc[0]], [ecutrho[0]])])
    for machine in machines:
        calc = batch.ParameterStudy(machine=machine, 
                                    templates=templates, 
                                    params_lst=params_lst, 
                                    prefix=prefix)
        calc.write_input(calc_dir=pj(testdir, 'calc_test'))
