import os
import numpy as np
from pwtools import io, parse,common
from pwtools.test.tools import assert_attrs_not_none
from pwtools.constants import Bohr,Ang,Ha,eV
from testenv import testdir

def test_cp2k_scf():
    attr_lst = parse.Cp2kSCFOutputFile().attr_lst
    for base in ['cp2k.scf.out.print_low', 'cp2k.scf.out.print_medium']:
        fn = 'files/cp2k/scf/%s' %base
        print "testing: %s" %fn
        print common.backtick("gunzip %s.gz" %fn)
        st = io.read_cp2k_scf(fn)
        assert_attrs_not_none(st, attr_lst=attr_lst)


def test_cp2k_md():
    attr_lst = parse.Cp2kMDOutputFile().attr_lst
    # This parser and others have get_econst(), but not all, so ATM it's not
    # part of the Trajectory API
    attr_lst.pop(attr_lst.index('econst'))
    for dr in ['files/cp2k/md/npt_f_print_low', 'files/cp2k/md/nvt_print_low']:
        base = os.path.dirname(dr) 
        fn = '%s/cp2k.out' %dr
        print "testing: %s" %fn
        print common.backtick('tar -C {0} -xzf {1}.tgz'.format(base,dr))
        tr = io.read_cp2k_md(fn)
        assert_attrs_not_none(tr, attr_lst=attr_lst)        
        pp = parse.Cp2kMDOutputFile(fn)
        forces_outfile = pp._get_forces_from_outfile()*Ha/Bohr/eV*Ang
        assert np.allclose(forces_outfile, tr.forces, rtol=1e-3)
        

def test_cp2k_cell_opt():
    attr_lst = parse.Cp2kRelaxOutputFile().attr_lst
    attr_lst.pop(attr_lst.index('econst'))
    # There is no PROJECT-frc-1.xyz file, but the input file has 
    #    &force_eval
    #        &print
    #            &forces
    #            &end forces
    #        &end print
    #    &end force_eval
    # therefore we can parse forces from the outfile.
    none_attrs = ['ekin',
                  'temperature',
                  'timestep',
                  'velocity',
                  ]
    for dr in ['files/cp2k/cell_opt/cell_opt']:
        base = os.path.dirname(dr) 
        fn = '%s/cp2k.out' %dr
        print "testing: %s" %fn
        print common.backtick('tar -C {0} -xzf {1}.tgz'.format(base,dr))
        tr = io.read_cp2k_relax(fn)
        assert_attrs_not_none(tr, attr_lst=attr_lst, none_attrs=none_attrs)        

