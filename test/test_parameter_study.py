import os
import numpy as np
from pwtools import comb, batch, common
from pwtools.pwscf import kpointstr_pwin

from testenv import testdir

def test():    
    pj = os.path.join

    calc_name = 'convergence'
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
                 comb.nested_loops([kpoints, zip([ecutwfc[0]], [ecutrho[0]])])
    for params in params_lst:
        print [(x.key, x.fileval) for x in common.flatten(params)]
    for machine in machines:
        templates[machine.jobfn] = batch.FileTemplate(basename=machine.jobfn,
                                                      templ_dir=templ_dir)
        calc = batch.ParameterStudy(machine=machine, 
                                    templates=templates, 
                                    params_lst=params_lst, 
                                    prefix=calc_name)
        calc.write_input(calc_dir=pj(testdir, 'calc_test'))
