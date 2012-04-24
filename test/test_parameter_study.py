# Parameter study examples. 
# 
# We use template files: files/calc.templ/* . These files are no real input
# files but do only contain placeholders in order to ease verification.
#
# We do not replace all placeholders in some examples, so do not wonder why the
# resulting "input files" sometimes still contain placeholders.
#
# There are actually 3 usage patterns:
#
# (1) Use 1 list per parameter + comb.nested_loops() to fill the param table:
#   get nested lists. Use sql.sql_matrix() with a header - the seldomly used
#   way.
# (2) Use 1 list per parameter, transform them with sql.sql_column(), then
#   use comb.nested_loops() with these lists - the common way.
# (3) Use direct loops + SQLEntry to fill the table "by hand" - the flexible
#   way.  

import os, shutil
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

local = batch.Machine(hostname='local',
                      subcmd='bash',
                      scratch='/tmp',
                      jobfn='job.local')

def assrt_aae(*args, **kwargs):
    np.testing.assert_array_almost_equal(*args, **kwargs)

def test():    
    pj = os.path.join
    templ_dir = 'files/calc.templ'
    
    #--------------------------------------------------------------------------
    # vary 1 parameter -- single loop
    #--------------------------------------------------------------------------
    # This may look overly complicated but in fact 50% of the code is only
    # verification. 
    #
    # database:
    # idx  ecutwfc .. some more cols ..
    # ---  -------
    # 0    25.0
    # 1    50.0
    # 2    75.0
    machine = local
    calc_dir = pj(testdir, 'calc_test_1d_1col')
    prefix = 'convergence'
    # Specify template files in templ_dir. Add machine template file to the
    # list of templates. The idea is to use this rather than spefifying it by
    # hand b/c the machine object may have things like "scratch" etc already
    # predefined. You can also loop over many machines here to write the same
    # input for several machines = [Machine(...), Machine(...), ...]. For that
    # to work, you must have machine.jobfn as template for each machine in
    # templ_dir.
    # for m in machines:
    #     templates.append(...)
    templates = [batch.FileTemplate(basename=fn, templ_dir=templ_dir) \
                 for fn in ['pw.in', machine.jobfn]]
    # raw values to be varied
    _ecutwfc = [25.0, 50.0, 75.0]
    # [SQLEntry(...), SQLEntry(...), SQLEntry(...)]
    ecutwfc = sql.sql_column(key='ecutwfc', lst=_ecutwfc)
    # only one parameter "ecutwfc" per calc (sublists of length 1) -> one sql
    # column. nested_loops(): transform
    # [SQLEntry(...), SQLEntry(...), SQLEntry(...)]
    # ->
    # [[SQLEntry(...)], # calc_dir/0
    #  [SQLEntry(...)], # calc_dir/1
    #  [SQLEntry(...)]] # calc_dir/2
    # or simply [[xx] for x in ecutfwc] 
    params_lst = comb.nested_loops([ecutwfc])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix,
                                calc_dir=calc_dir)
    calc.write_input()
    # asserts .......
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    header = [(x[0].lower(), x[1].lower()) for x in db.get_header()]
    assert ('idx', 'integer') in header
    assert ('ecutwfc', 'real') in header
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
    # Incomplete parameter sets I and zip().
    # Add more ecutwfc to the same study + one column of misc information. Vary
    # two parameters (ecutwfc, pw_mkl) *together* using the zip() trick.
    #--------------------------------------------------------------------------

    # database:
    # idx  ecutwfc pw_mkl  .. some more cols ..
    # ---  ------- ------
    # 0    25.0    
    # 1    50.0
    # 2    75.0
    # 3    100.0   'yes'
    # 4    150.0   'yes'
    #
    # Empty fields for idx=0,1,2 in col `pw_mkl` are NULL (sqlite type), like
    # None in Python. This is a simple example of "incomplete parameter sets".
    #
    ecutwfc = sql.sql_column(key='ecutwfc', lst=[100.0, 150.0])
    pw_mkl = sql.sql_column(key='pw_mkl',  lst=['yes', 'yes'])
    params_lst = comb.nested_loops([zip(ecutwfc, pw_mkl)], flatten=True)
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix,
                                calc_dir=calc_dir)
    calc.write_input()
    # asserts ...
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    header = [(x[0].lower(), x[1].lower()) for x in db.get_header()]
    assert ('pw_mkl', 'text') in header
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2,3,4]
    assert db.get_list1d("select pw_mkl from calc where idx <=2") == [None]*3
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
    # Repeat first test, but whith templates = dict, w/o verification though
    #--------------------------------------------------------------------------
    #
    # This tests backwd compat API where the `templates` arg to ParameterStudy
    # can be a dict, too. Nowadays, we use lists instead.
    machine = local
    calc_dir = pj(testdir, 'calc_test_1d_1col_templdict')
    prefix = 'convergence'
    templates = \
        {'pw': batch.FileTemplate(basename='pw.in', 
                                  templ_dir=templ_dir), 
        }
    templates[machine.jobfn] = batch.FileTemplate(basename=machine.jobfn,
                                                  templ_dir=templ_dir)

    _ecutwfc = [25.0, 50.0, 75.0]
    ecutwfc = sql.sql_column(key='ecutwfc', lst=_ecutwfc)
    params_lst = comb.nested_loops([ecutwfc])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix,
                                calc_dir=calc_dir)
    calc.write_input()

    #--------------------------------------------------------------------------
    # Incomplete parameter sets II
    #--------------------------------------------------------------------------

    # Use this to fill the parameter table with arbitrarily complex patterns by
    # explicit loops and direct SQLEntry construction. This makes sense if you
    # have many placeolders where only certain combinations make sense and you
    # don't want to invent default values for the others to "fill the gaps": In
    # the example below, we have 10 columns: subcmd, hostname, conv_thr, scratch,
    # kpoints, idx, jobfn, prefix, home, ecutwfc (some come from
    # batch.<machine>). Normally, one would have to fill each entry in each row
    # i.e. construct a full table. This is done by each params_lst sublist
    # beeing a list of length 10 (i.e. for each parameter). This is not
    # necessary with "incomplete parameter sets". For each row (= Calculation),
    # just set the paeameters which wou want to vary. The others are NULL by
    # default.
    # 
    # Note that empty fields are NULL in sqlite and None in Python. Do queries
    # with
    #   "select * from calc where ecutwfc IS NULL"
    # The syntax "... ecutfwc==NULL" is wrong.  
    #
    # database:
    # subcmd      hostname        conv_thr    scratch     kpoints     idx         jobfn       prefix            home             ecutwfc   
    # ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------------  ---------------  ----------
    # bash        local                   /tmp                    0           job.local   convergence_run0  /home/schmerler  25.0      
    # bash        local                   /tmp                    1           job.local   convergence_run1  /home/schmerler  50.0      
    # bash        local                   /tmp        2 2 2 0 0   2           job.local   convergence_run2  /home/schmerler            
    # bash        local                   /tmp        4 4 4 0 0   3           job.local   convergence_run3  /home/schmerler            
    # bash        local       1.0e-08     /tmp        6 6 6 0 0   4           job.local   convergence_run4  /home/schmerler  75.0    

    machine = local
    # make sure that scratch == '/tmp'
    machine.scratch = '/tmp'
    calc_dir = pj(testdir, 'calc_test_incomplete')
    prefix = 'convergence'
    templates = [batch.FileTemplate(basename=fn, templ_dir=templ_dir) \
                 for fn in ['pw.in', machine.jobfn]]
    params = []
    # Row with 1 column:
    # The first two rows (=parameter sets) vary only "ecutwfc". No default
    # values for the other columns have to be invented. They are simply NULL.
    #   [[SQLEntry(...,25)],
    #    [SQLEntry(...,50)]]
    for xx in [25.0,50.0]:
        params.append([sql.SQLEntry(key='ecutwfc', sqlval=xx)])
    # Row with 1 column:    
    # Row 2 and 3 vary only kpoints, leaving ecutwfc=NULL this time.
    for xx in ['2 2 2 0 0 0', '4 4 4 0 0 0']:
        params.append([sql.SQLEntry(key='kpoints', sqlval=xx)])
    # Now, one row with 3 columns. You can add parameters (column names)
    # whenever you want. They are appended to the bottom if the table. Default
    # values for erlier rows are NULL.
    params.append([sql.SQLEntry(key='ecutwfc', sqlval=75.0),
                   sql.SQLEntry(key='kpoints', sqlval='6 6 6 0 0 0'),
                   sql.SQLEntry(key='conv_thr', sqlval=1e-8)])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params, 
                                prefix=prefix,
                                calc_dir=calc_dir)
    calc.write_input()
    # asseerts ...
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    assert db.get_list1d('select conv_thr from calc') == [None]*4 +[1e-8]
    assert db.get_list1d('select ecutwfc from calc') == [25.0, 50.0, None, None, 75.0]
    assert db.get_list1d('select scratch from calc') == ['/tmp']*5
    # some columns actually depend on batch.Machine
    hdr = [('subcmd', 'TEXT'),  # machine
           ('hostname', 'TEXT'),    # machine
           ('conv_thr', 'REAL'), 
           ('scratch', 'TEXT'), 
           ('kpoints', 'TEXT'), 
           ('idx', 'INTEGER'),  # automatic by ParameterStudy
           ('jobfn', 'TEXT'),   # machine
           ('prefix', 'TEXT'),  # automatic by ParameterStudy
           ('ecutwfc', 'REAL')]
    db_hdr = db.get_header()           
    for colspec in hdr:
        assert colspec in db_hdr

    #--------------------------------------------------------------------------
    # check some defaults, don't write input data
    #--------------------------------------------------------------------------
    machine = local
    prefix = 'convergence'
    templates = [batch.FileTemplate(basename=fn, templ_dir=templ_dir) \
                 for fn in ['pw.in', machine.jobfn]]
    _ecutwfc = [25.0, 50.0, 75.0]
    ecutwfc = sql.sql_column(key='ecutwfc', lst=_ecutwfc)
    params_lst = comb.nested_loops([ecutwfc])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix='foo')
    assert calc.calc_dir == pj(os.curdir, 'calc_%s' % machine.hostname)

    #--------------------------------------------------------------------------
    # check mode
    #--------------------------------------------------------------------------
    calc_dir = pj(testdir, 'calc_test_1d_mode')
    machine = local
    prefix = 'convergence'
    templates = [batch.FileTemplate(basename=fn, templ_dir=templ_dir) \
                 for fn in ['pw.in', machine.jobfn]]
    _ecutwfc = [1.0, 2.0, 3.0]
    ecutwfc = sql.sql_column(key='ecutwfc', lst=_ecutwfc)
    params_lst = comb.nested_loops([ecutwfc])
    calc = batch.ParameterStudy(machine=machine, 
                                templates=templates, 
                                params_lst=params_lst, 
                                prefix=prefix,
                                calc_dir=calc_dir)
    # ----- append ----------
    calc.write_input(mode='a') # or 'w'
    # asserts .......
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2]
    assrt_aae(db.get_array1d("select ecutwfc from calc"), 
              np.array([1.0, 2.0, 3.0]))

    calc.write_input(mode='a', backup=False)
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2,3,4,5]
    assrt_aae(db.get_array1d("select ecutwfc from calc"), 
              np.array([1.0, 2.0, 3.0]*2))
    
    # ----- write ----------
    shutil.rmtree(calc_dir)
    calc.write_input(mode='a') # or 'w'
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2]
    db.add_column('foo', 'integer')
    for idx in idx_lst:
        subdir = pj(calc_dir, str(idx))
        common.system("touch %s/foo" %subdir)
        db.execute("update calc set foo=? where idx==?", (idx*100, idx))
    print db.execute("select * from calc").fetchall()        
    db.finish()        
    calc.write_input(mode='w', backup=True)
    db = sql.SQLiteDB(pj(calc_dir, 'calc.db'), table='calc')
    idx_lst = db.get_list1d("select idx from calc")
    assert idx_lst == [0,1,2]
    assert not db.has_column('foo')
    # files 'foo' exist only in backup
    for idx in idx_lst:
        assert not os.path.exists(pj(calc_dir, str(idx), 'foo'))
        assert os.path.exists(pj(calc_dir + '.0', str(idx), 'foo'))

