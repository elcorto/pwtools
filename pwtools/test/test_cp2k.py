import os
import numpy as np
from pwtools import io, parse,common
from pwtools.test.tools import assert_attrs_not_none
from pwtools.constants import Bohr,Ang,Ha,eV
from .testenv import testdir
from pwtools.test.tools import unpack_compressed
pj = common.pj

def test_cp2k_scf():
    attr_lst = parse.Cp2kSCFOutputFile().attr_lst
    for base in ['cp2k.scf.out.print_low', 'cp2k.scf.out.print_medium']:
        fn = 'files/cp2k/scf/%s' %base
        print("testing: %s" %fn)
        print(common.backtick("gunzip %s.gz" %fn))
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
        print("testing: %s" %fn)
        print(common.backtick('tar -C {0} -xzf {1}.tgz'.format(base,dr)))
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
        print("testing: %s" %fn)
        print(common.backtick('tar -C {0} -xzf {1}.tgz'.format(base,dr)))
        tr = io.read_cp2k_relax(fn)
        assert_attrs_not_none(tr, attr_lst=attr_lst, none_attrs=none_attrs)


def test_cp2k_txt_vs_dcd():
    # Two exactly equal NPT runs, nstep=16, natoms=57. The dcd run uses
    #   motion/print/trajectory format dcd_aligned_cell
    # the other the default xyz format. So we have
    #   PROJECT-pos-1.dcd
    #   PROJECT-pos-1.xyz
    # Since the cell changes and we use dcd_aligned_cell, the coords from the
    # xyz run are NOT the same as the coords in the dcd file, which HAS to be
    # like this. Only coords_frac can be compared, and coords after the xyz
    # run's cell has been aligned to [[x,0,0],[xy,y,0],[xz,yz,z]].
    dir_xyz = unpack_compressed('files/cp2k/dcd/npt_xyz.tgz')
    dir_dcd = unpack_compressed('files/cp2k/dcd/npt_dcd.tgz')
    tr_xyz = io.read_cp2k_md(pj(dir_xyz, 'cp2k.out'))
    tr_dcd = io.read_cp2k_md_dcd(pj(dir_dcd, 'cp2k.out'))
    for name in ['natoms', 'nstep', 'timestep', 'symbols']:
        assert getattr(tr_xyz,name) == getattr(tr_dcd, name)
    assert tr_xyz.timestep == 1.0
    assert tr_xyz.natoms == 57
    assert tr_xyz.nstep == 16

    # coords are 32bit float in dcd files (single prec, so coords_frac can only
    # be accurate to that precision, which ~1e-8). cryst_const is double
    # precision in the dcd file, so atol can be lower
    assert np.allclose(tr_xyz.coords_frac, tr_dcd.coords_frac, rtol=0,
                       atol=1.5e-7)
    assert np.allclose(tr_xyz.cryst_const, tr_dcd.cryst_const, rtol=0,
                       atol=7e-10)
    assert not np.allclose(tr_xyz.coords, tr_dcd.coords, rtol=0,
                           atol=5e-6)
    assert not np.allclose(tr_xyz.cell, tr_dcd.cell, rtol=0,
                           atol=7e-10)

    # align xyz cell, now cell and coords are the same
    tr_xyz.coords = None; tr_xyz.cell=None; tr_xyz.set_all()

    assert np.allclose(tr_xyz.coords_frac, tr_dcd.coords_frac, rtol=0,
                       atol=1.5e-7)
    assert np.allclose(tr_xyz.cryst_const, tr_dcd.cryst_const, rtol=0,
                       atol=7e-10)
    assert np.allclose(tr_xyz.coords, tr_dcd.coords, rtol=0,
                       atol=5e-6)
    assert np.allclose(tr_xyz.cell, tr_dcd.cell, rtol=0,
                       atol=7e-10)

