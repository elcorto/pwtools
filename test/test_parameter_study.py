# Parameter study examples. 
# 
# We use template files: files/calc.templ/* . These files are no real input
# files but do only contain placeholders in order to ease verification.
#
# We do not replace all placeholders in some examples, so do not wonder why the
# resulting "input files" sometimes still contain placeholders.

import os
import numpy as np
from pwtools import comb, batch, common
from pwtools.pwscf import kpointstr_pwin
from pwtools import sql
from testenv import testdir

def file_get(fn, key):
    # parse "key=value" lines, return "value" of first key found as string
    lines = common.file_readlines(fn)
    for ll in lines:
        if ll.strip().startswith(key):
            return ll.split('=')[1].strip()

def test():    
    pj = os.path.join
    templ_dir = 'files/calc.templ'
    
    #--------------------------------------------------------------------------
    # vary 1 parameter -- single loop
    #--------------------------------------------------------------------------
    # This may look overly complicated but in fact 50% of the code is only
    # verification.
    machine = batch.local
    calc_dir = pj(testdir, 'calc_test_1d_1col')
    prefix = 'convergence'
    # Specify template files in templ_dir. Add machine template file to the
    # list of templates. The idea is to use this rather than spefifying it by
    # hand b/c the machine object may have things like "scratch" etc already
    # predefined. You can also loop over many machines here to write the same
    # input for several machines = [batch.local, batch.adde, ...]. For that to
    # work, you must have machine.jobfn as template for each machine in
    # templ_dir.
    # for m in machines:
    #     templates.append(...)
    templates = [batch.FileTemplate(basename=fn, templ_dir=templ_dir) \
                 for fn in ['pw.in', machine.jobfn]]
    # raw values to be varied
    _ecutwfc = [25, 50, 75]
    # [SQLEntry(...), SQLEntry(...), SQLEntry(...)]
    ecutwfc = batch.sql_column('ecutwfc', 'float', _ecutwfc)
    # only one parameter "ecutwfc" per calc (sublists of length 1) -> one sql
    # column. nested_loops(): transform
    # [SQLEntry(...), SQLEntry(...), SQLEntry(...)]
    # ->
    # [[SQLEntry(...)], # calc_dir/0
    #  [SQLEntry(...)], # calc_dir/1
    #  [SQLEntry(...)]] # calc_dir/2
    params_lst = comb.nested_loops([ecutwfc])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix)
    calc.write_input(calc_dir=calc_dir)
    # asserts .......
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    header = [(x[0].lower(), x[1].lower()) for x in db.get_header()]
    assert ('idx', 'integer') in header
    assert ('ecutwfc', 'float') in header
    assert ('scratch', 'text') in header
    assert ('prefix', 'text') in header
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2]
    for idx in idx_lst:
        pwfn = pj(calc_dir, str(idx), 'pw.in')
        jobfn = pj(calc_dir, str(idx), machine.jobfn) 
        assert float(ecutwfc[idx].fileval) == float(file_get(pwfn, 'ecutwfc'))
        assert float(ecutwfc[idx].fileval) == float(_ecutwfc[idx])
        # same placeholders in different files
        assert machine.scratch == file_get(jobfn, 'scratch')
        assert machine.scratch == file_get(pwfn, 'outdir')
        prfx = prefix + '_run%i' %idx
        # same placeholders in different files
        assert prfx == file_get(pwfn, 'prefix')
        assert prfx == file_get(jobfn, 'prefix')
        # content of sqlite database
        assert prfx == db.execute("select prefix from calc where idx==?",
                                  (idx,)).fetchone()[0]
        assert _ecutwfc[idx] == db.execute("select ecutwfc from calc where idx==?",
                                           (idx,)).fetchone()[0]
        assert machine.scratch == db.execute("select scratch from calc where idx==?",
                                             (idx,)).fetchone()[0]
    
    
    #--------------------------------------------------------------------------
    # Add more ecutwfc to the same study + one column of misc information. Vary
    # two parameters (ecutwfc, pw_mkl) *together* using the zip() trick.
    #--------------------------------------------------------------------------
    ecutwfc = batch.sql_column('ecutwfc', 'float', [100, 150])
    pw_mkl = batch.sql_column('pw_mkl', 'text', ['yes', 'yes'])
    params_lst = comb.nested_loops([zip(ecutwfc, pw_mkl)], flatten=True)
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix)
    calc.write_input(calc_dir=calc_dir)
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    header = [(x[0].lower(), x[1].lower()) for x in db.get_header()]
    assert ('pw_mkl', 'text') in header
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2,3,4]
    for idx, ecut in zip([3,4], [100, 150]):
        pwfn = pj(calc_dir, str(idx), 'pw.in')
        jobfn = pj(calc_dir, str(idx), machine.jobfn)
        # the study index is 3,4, but the local parameter index is 0,1
        assert float(ecutwfc[idx-3].fileval) == \
               float(file_get(pwfn, 'ecutwfc'))
        assert float(ecut) == float(file_get(pwfn, 'ecutwfc'))
        assert prefix + '_run%i' %idx == file_get(pwfn, 'prefix')
        assert prefix + '_run%i' %idx == file_get(jobfn, 'prefix')

    #--------------------------------------------------------------------------
    # Vary two (three, ...) params on a 2d (3d, ...) grid
    #--------------------------------------------------------------------------
    # This is more smth for the examples/, not tests. In fact, the way you are
    # constructing params_lst is only a matter of zip() and comb.nested_loops().
     
    # >>> par1 = batch.sql_column('par1', 'float', [1,2,3])
    # >>> par2 = batch.sql_column('par2', 'text', ['a','b'])
    # >>> par3 = ...
    #
    # # 2d grid
    # >>> params_lst = comb.nested_loops([par1, par2])
    # 
    # # 3d grid   
    # >>> params_lst = comb.nested_loops([par1, par2, par3])
    # 
    # # vary par1 and par2 together, and par3 -> 2d grid w/ par1+par2 on one
    # axis and par3 on the other
    # >>> params_lst = comb.nested_loops([zip(par1, par2), par3], flatten=True)
    #
    # That's all.
    
    #--------------------------------------------------------------------------
    # Repeat first test, but whith templates = dict, w/o verification though
    #--------------------------------------------------------------------------
    machine = batch.local
    calc_dir = pj(testdir, 'calc_test_1d_1col_templdict')
    prefix = 'convergence'
    templates = \
        {'pw': batch.FileTemplate(basename='pw.in', 
                                  templ_dir=templ_dir), 
        }
    templates[machine.jobfn] = batch.FileTemplate(basename=machine.jobfn,
                                                  templ_dir=templ_dir)

    _ecutwfc = [25, 50, 75]
    ecutwfc = batch.sql_column('ecutwfc', 'float', _ecutwfc)
    params_lst = comb.nested_loops([ecutwfc])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix)
    calc.write_input(calc_dir=calc_dir)

