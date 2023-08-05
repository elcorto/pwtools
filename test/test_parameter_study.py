import os
import numpy as np
from pwtools import comb, batch, common, sql
from pwtools.test.tools import all_types_equal, assert_all_types_equal
from pwtools.test.testenv import testdir
pj = os.path.join

def check_key_in_file(lines, key, file_target):
    """If line "key=<value>" is found in file, then convert the string
    containing <value> to Python type and assert value==file_target.
    """
    for ll in lines:
        if ll.strip().startswith(key):
            file_val_str = ll.split('=')[1].strip()
            print("check_key_in_file: key={0}, "
                  "file_val_str={1}, file_target={2}".format(key, file_val_str, file_target))
            # hack to convert string from file to correct type, failed
            # conversion raises ValueError
            ret = False
            for converter in [repr, str, int, float]:
                try:
                    file_val = converter(file_val_str)
                    ok = True
                except ValueError:
                    ok = False
                if ok:
                    ret = all_types_equal(file_target, file_val)
                    if ret:
                        break
            assert ret, ("not equal: file_target={}, "
                         "file_val={}".format(file_target, file_val))


def check_generated(calc_root, machine_dct, params_lst, revision):
    """Check consistency of calc database values, replacement params in
    `params_lst` and all written files.
    """
    dbfn = pj(calc_root, 'calc.db')
    db = sql.SQLiteDB(dbfn, table='calc')
    print("database content:")
    print(db.get_dict("select * from calc"))
    db_colnames = [x[0] for x in db.get_header()]
    for idx,hostname_str in db.execute("select idx,hostname from calc \
                                        where revision==?", (revision,)).fetchall():
        for hostname in hostname_str.split(','):
            machine = machine_dct[hostname]
            calc_dir = pj(calc_root, 'calc_%s' %machine.hostname, str(idx))
            for base in ['pw.in', machine.get_jobfile_basename()]:
                fn = pj(calc_dir, base)
                assert os.path.exists(fn)
                lines = common.file_readlines(fn)
                # assemble all possible replacements in one list of SQLEntry
                # instances, some things are redundantely checked twice ...
                sql_lst = params_lst[idx] + list(machine.get_sql_record().values())
                for db_key in db_colnames:
                    db_val = db.get_single("select %s from calc "
                                           "where idx==?" %db_key,
                                           (idx,))
                    if db_val is not None:
                        sql_lst.append(sql.SQLEntry(key=db_key, sqlval=db_val))
                # for each replacement key, check if they are correctly placed
                # in the database (if applicable) and in the written files
                for sqlentry in sql_lst:
                    if sqlentry.key in db_colnames:
                        db_val = db.get_single("select %s from calc "
                                               "where idx==?" \
                                               %sqlentry.key, (idx,))
                        assert_all_types_equal(db_val, sqlentry.sqlval)
                    else:
                        db_val = 'NOT_DEFINED_IN_DB'
                    print("check_generated: idx={0}, sqlentry.key={1}, "
                          "sqlentry.sqlval={2}, db_val={3}".format(idx, sqlentry.key,
                                                                 sqlentry.sqlval,
                                                                 db_val))
                    check_key_in_file(lines, sqlentry.key, sqlentry.sqlval)
    db.finish()


def test_parameter_study():
    templ_dir = 'files/calc.templ'
    calc_root = pj(testdir, 'calc_test_param_study')

    # filename: FileTemplate built from that internally
    host0 = batch.Machine(hostname='host0',
                          subcmd='qsub_host0',
                          home='/home/host0/user',
                          scratch='/tmp/host0',
                          filename='files/calc.templ/job.host0')

    # template: provide FileTemplate directly
    host1 = batch.Machine(hostname='host1',
                          subcmd='qsub_host1',
                          home='/home/host1/user',
                          scratch='/tmp/host1',
                          template=batch.FileTemplate(basename='job.host1',
                                                      templ_dir=templ_dir))

    # use template text here instead of a file
    host2_txt = """
subcmd=XXXSUBCMD
scratch=XXXSCRATCH
home=XXXHOME
calc_name=XXXCALC_NAME
idx=XXXIDX
revision=XXXREVISION
study_name=XXXSTUDY_NAME
"""
    host2 = batch.Machine(hostname='host2',
                          subcmd='qsub_host2',
                          home='/home/host2/user',
                          scratch='/tmp/host2',
                          template=batch.FileTemplate(basename='job.host2',
                                                      txt=host2_txt))



    study_name = 'convergence'
    templates = [batch.FileTemplate(basename='pw.in', templ_dir=templ_dir)]
    param0 = sql.sql_column(key='param0', lst=[25.0, 50.0])
    param1 = sql.sql_column(key='param1', lst=['2x2x2','3x3x3','4x4x4'])
    param2 = sql.sql_column(key='param2', lst=[77,88,99,111])

    # only needed for this test
    machine_dct = {'host0': host0,
                   'host1': host1,
                   'host2': host2,
                   }
    nparam0 = len(param0)
    nparam1 = len(param1)
    nparam2 = len(param2)

    #------------------------------------------------------------------------
    # revision=0
    #------------------------------------------------------------------------
    params_lst0 = comb.nested_loops([param0])
    calc = batch.ParameterStudy(machines=host0,
                                templates=templates,
                                params_lst=params_lst0,
                                study_name=study_name,
                                calc_root=calc_root)
    # same as mode='w' + backup=True
    calc.write_input(mode='a', backup=True)
    check_generated(calc_root, machine_dct, params_lst0, revision=0)

    #------------------------------------------------------------------------
    # revision=0, no backup, erase all
    #------------------------------------------------------------------------
    params_lst0 = comb.nested_loops([param0])
    calc = batch.ParameterStudy(machines=host0,
                                templates=templates,
                                params_lst=params_lst0,
                                study_name=study_name,
                                calc_root=calc_root)
    calc.write_input(mode='w', backup=False)
    check_generated(calc_root, machine_dct, params_lst0, revision=0)
    assert not os.path.exists(pj(calc_root, 'calc_host0.0'))
    assert not os.path.exists(pj(calc_root, 'calc.db.0'))
    # only calc_foo/0 ... calc_foo/{N-1}
    for ii in range(nparam0):
        assert os.path.exists(pj(calc_root, 'calc_host0/%i' %ii))
    for jj in range(1,5):
        assert not os.path.exists(pj(calc_root, 'calc_host0/%i' %(ii+jj,)))

    #------------------------------------------------------------------------
    # revision=0, backup, then erase all
    #------------------------------------------------------------------------
    params_lst0 = comb.nested_loops([param0])
    calc = batch.ParameterStudy(machines=host0,
                                templates=templates,
                                params_lst=params_lst0,
                                study_name=study_name,
                                calc_root=calc_root)
    calc.write_input(mode='w', backup=True)
    check_generated(calc_root, machine_dct, params_lst0, revision=0)
    assert os.path.exists(pj(calc_root, 'calc_host0.0'))
    assert os.path.exists(pj(calc_root, 'calc.db.0'))
    # only calc_foo/0 ... calc_foo/{N-1}
    for ii in range(nparam0):
        assert os.path.exists(pj(calc_root, 'calc_host0/%i' %ii))
    for jj in range(1,5):
        assert not os.path.exists(pj(calc_root, 'calc_host0/%i' %(ii+jj,)))

    #------------------------------------------------------------------------
    # revision=1, backup and extend
    #------------------------------------------------------------------------
    params_lst1 = comb.nested_loops([param1,param2])
    calc = batch.ParameterStudy(machines=[host0,host1,host2],
                                templates=templates,
                                params_lst=params_lst1,
                                study_name=study_name,
                                calc_root=calc_root)
    calc.write_input(mode='a', backup=True)
    assert os.path.exists(pj(calc_root, 'calc_host0.1'))
    assert os.path.exists(pj(calc_root, 'calc.db.1'))
    for ii in range(nparam0 + nparam1*nparam2):
        assert os.path.exists(pj(calc_root, 'calc_host0/%i' %ii))
    for ii in range(nparam0):
        assert not os.path.exists(pj(calc_root, 'calc_host1/%i' %ii))
        assert not os.path.exists(pj(calc_root, 'calc_host2/%i' %ii))
    for ii in range(nparam0+1, nparam1*nparam2):
        assert os.path.exists(pj(calc_root, 'calc_host1/%i' %ii))
        assert os.path.exists(pj(calc_root, 'calc_host2/%i' %ii))

    # excl_push
    excl_fn = pj(calc_root, 'excl_push')
    # ['0', '1', '2', ...]
    assert common.file_read(excl_fn).split() == \
        [str(x) for x in range(len(params_lst0))]

    # sum params_lstm b/c we use `idx` from calc.db and that counts params_lst0
    # + params_lst1, i.e. all paramseter sets from revision=0 up to now
    check_generated(calc_root, machine_dct, params_lst0+params_lst1, revision=1)


def test_default_repl_keys():
    batch.default_repl_keys()

