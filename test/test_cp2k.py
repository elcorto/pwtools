import os
from pwtools import io, parse,common
from pwtools.test.tools import assert_attrs_not_none
from testenv import testdir

def test_cp2k_scf():
    attr_lst = parse.Cp2kSCFOutputFile().attr_lst
    for base in ['cp2k.scf.out.print_low', 'cp2k.scf.out.print_medium']:
        fn = 'files/cp2k/scf/%s' %base
        print common.backtick("gunzip %s.gz" %fn)
        st = io.read_cp2k_scf(fn)
        assert_attrs_not_none(st, attr_lst=attr_lst)

def test_cp2k_md():
    attr_lst = parse.Cp2kMDOutputFile().attr_lst
    # This parser and others have get_econst(), but not all, so ATM it's not
    # part of the Trajectory API
    attr_lst.pop(attr_lst.index('econst'))
    dr = 'files/cp2k/md/print_level_low'
    base = os.path.dirname(dr) 
    fn = '%s/cp2k.out' %dr
    print common.backtick('tar -C {} -xzf {}.tgz'.format(base,dr))
    tr = io.read_cp2k_md(fn)
    assert_attrs_not_none(tr, attr_lst=attr_lst)        
