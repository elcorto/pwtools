#!/usr/bin/python

import numpy as np
from pwtools import comb, sql, batch, common, parse, constants, crys
str_arr = common.str_arr

if __name__ == '__main__':
    
    prefix = 'md'
    machines = [batch.local]
    templates = \
        {'abiin': batch.FileTemplate(basename='abi.in'), 
         'abifiles': batch.FileTemplate(basename='abi.files'), 
        } 
    for m in machines:
        templates[m.jobfn] = batch.FileTemplate(basename=m.jobfn)
    
    #-------------------------------------------------------------------------
    _ionmov  = [2,    2,      2,     8,    13,   13,     13]
    _optcell = [0,    1,      2,     0,     0,    1,      2]
    _dilatmx = [1,    1.1,    1.1,   1,     1,    1.1,    1.1]
    _ecutsm  = [0,    0.3,    0.3,   0,     0,    0.3,    0.3]
    
    ionmov = batch.sql_column('ionmov', 'integer', _ionmov)
    optcell = batch.sql_column('optcell', 'integer', _optcell)
    dilatmx = batch.sql_column('dilatmx', 'float', _dilatmx)
    ecutsm = batch.sql_column('ecutsm', 'float', _ecutsm)
    
    #-------------------------------------------------------------------------

    params_lst = comb.nested_loops([zip(ionmov, optcell, dilatmx, ecutsm)], 
                                   flatten=True)

    for machine in machines:
        calc = batch.ParameterStudy(machine=machine, 
                                    templates=templates, 
                                    params_lst=params_lst, 
                                    prefix=prefix)
        calc.write_input(sleep=0, backup=False)
