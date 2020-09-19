import os, time
import numpy as np

import pytest

from pwtools import io, constants, common, parse
from pwtools.calculators import Pwscf, Lammps, find_exe
from .testenv import testdir

pj = os.path.join

prefix = 'ase_calculator'
lammps_exe_names = ['lammps', 'lmp']


def have_ase():
    try:
        import ase
        return True
    except ImportError:
        return False


def have_pwx():
    return find_exe('pw.x') is not None


def have_lmp():
    return find_exe(lammps_exe_names) is not None


def get_atoms_with_calc_pwscf(pseudo_dir):
    st_start = io.read_pw_scf('files/ase/pw.scf.out.start.rs-AlN')
    at = st_start.get_ase_atoms(pbc=True)
    calc = Pwscf(label='pw',
                 directory=pj(testdir, prefix, 'calc_dir'),
                 kpts=1/0.9,
                 ecutwfc=30,
                 conv_thr=1e-5,
                 pp='pbe-n-kjpaw_psl.0.1.UPF',
                 pseudo_dir=pseudo_dir,
                 calc_name='my_calc',
                 outdir=pj(testdir, prefix, 'scratch'),
                 command='pw.x < pw.in > pw.out 2>&1',
                 backup=True,
                 )
    at.set_calculator(calc)
    return at


def get_atoms_with_calc_lammps():
    st_start = io.read_pw_scf('files/ase/pw.scf.out.start.wz-AlN')
    at = st_start.get_ase_atoms(pbc=True)
    exe = find_exe(lammps_exe_names)
    assert exe is not None
    calc = Lammps(label='lmp',
                  directory=pj(testdir, prefix, 'calc_dir'),
                  pair_style='tersoff',
                  pair_coeff='* * AlN.tersoff Al N',
                  command=f'{exe} < lmp.in > lmp.out 2>&1',
                  backup=True,
                  )
    at.set_calculator(calc)
    return at


def setup_pwscf():
    pseudo_dir = pj(testdir, prefix, 'pseudo')
    if not os.path.exists(pseudo_dir):
        common.backtick(f"mkdir -pv {pseudo_dir}; "
                        f"cp files/qe_pseudos/*.gz {pseudo_dir}/; "
                        f"gunzip {pseudo_dir}/*.gz")
    return get_atoms_with_calc_pwscf(pseudo_dir)


def setup_lmp():
    at = get_atoms_with_calc_lammps()
    at.rattle(stdev=0.001, seed=int(time.time()))
    common.makedirs(at.calc.directory)
    common.backtick(f"cp -v utils/lammps/AlN.tersoff {at.calc.directory}/")
    return at


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_pwx()")
def test_pwscf_calculator_scf():
    at = setup_pwscf()
    # trigger calculation here
    forces = at.get_forces()
    etot = at.get_potential_energy()
    stress = at.get_stress(voigt=False) # 3x3

    st = io.read_pw_scf(at.calc.label + '.out')
    assert np.allclose(forces, st.forces)
    assert np.allclose(etot, st.etot)
    assert np.allclose(st.stress, -stress * constants.eV_by_Ang3_to_GPa)


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_pwx()")
def test_pwscf_calculator_vc_relax():
    at = setup_pwscf()
    # files/ase/pw.scf.out.start is a norm-conserving LDA struct,
    # calculated with pz-vbc.UPF, so the PBE vc-relax will make the cell
    # a bit bigger
    from ase.optimize import BFGS
    from ase.constraints import UnitCellFilter
    traj_fn = f"{at.calc.directory}/test_pwscf_calculator_vc_relax.traj"
    os.makedirs(os.path.dirname(traj_fn), exist_ok=True)
    opt = BFGS(UnitCellFilter(at),
               trajectory=traj_fn)
    cell = parse.arr2d_from_txt("""
        -1.97281509  0.          1.97281509
         0.          1.97281509  1.97281509
        -1.97281509  1.97281509  0.""")
    assert np.allclose(cell, at.get_cell())
    opt.run(fmax=0.05) # run only few steps
    cell = parse.arr2d_from_txt("""
        -2.01841537  0           2.01841537
        0            2.01841537  2.01841537
        -2.01841537  2.01841537  0""")
    assert np.allclose(cell, at.get_cell())

    # at least 1 backup files must exist: pw.*.0 is the SCF run, backed up
    # in the first iter of the vc-relax
    assert os.path.exists(at.calc.infile + '.0')


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_lmp()")
def test_lammps_calculator_single_point():
    at = setup_lmp()
    forces = at.get_forces()
    etot = at.get_potential_energy()
    stress = at.get_stress(voigt=False) # 3x3

    st = io.read_lammps_md_txt(at.calc.label + '.out')[0]
    assert np.allclose(forces, st.forces)
    assert np.allclose(etot, st.etot)
    assert np.allclose(st.stress, -stress * constants.eV_by_Ang3_to_GPa,
                       atol=1e-10)


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_lmp()")
def test_lammps_calculator_relax():
    at = setup_lmp()
    from ase.optimize import BFGS
    opt = BFGS(at, maxstep=0.04)
    opt.run(fmax=0.001, steps=10)
    coords_frac = parse.arr2d_from_txt("""
        3.3333341909920072e-01    6.6666683819841532e-01    4.4325467247779138e-03
        6.6666681184103216e-01    3.3333362368205072e-01    5.0443254824788963e-01
        3.3333341909918301e-01    6.6666683819838046e-01    3.8356759709402671e-01
        6.6666681184101539e-01    3.3333362368201563e-01    8.8356759861713752e-01
        """)
    assert np.allclose(coords_frac, at.get_scaled_positions(), atol=1e-2)

    # at least 1 backup files must exist
    assert os.path.exists(at.calc.infile + '.0')
    assert os.path.exists(at.calc.outfile + '.0')
    assert os.path.exists(at.calc.dumpfile + '.0')
    assert os.path.exists(at.calc.structfile + '.0')
