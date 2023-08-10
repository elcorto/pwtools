import os
import time
from importlib.util import find_spec

import numpy as np

import pytest

from pwtools import io, constants, common, parse, crys
from pwtools.calculators import Pwscf, Lammps, find_exe

pj = os.path.join

lammps_exe_names = ["lammps", "lmp"]
here = common.fullpath(os.path.dirname(__file__))


def have_ase():
    return find_spec("ase") is not None


def have_pwx():
    return find_exe("pw.x") is not None


def have_lmp():
    return find_exe(lammps_exe_names) is not None


def get_atoms_with_calc_pwscf(pseudo_dir, tmpdir):
    st_start = io.read_pw_scf("files/ase/pw.scf.out.start.rs-AlN")
    at = st_start.get_ase_atoms(pbc=True)
    calc = Pwscf(
        label="pw",
        directory=tmpdir / "calc_dir",
        kpts=1 / 0.9,
        ecutwfc=30,
        conv_thr=1e-5,
        pp="pbe-n-kjpaw_psl.0.1.UPF",
        pseudo_dir=pseudo_dir,
        calc_name="my_calc",
        outdir=tmpdir / "scratch",
        command="pw.x < pw.in > pw.out 2>&1",
        backup=True,
    )
    at.set_calculator(calc)
    return at


def get_atoms_with_calc_lammps(tmpdir):
    st_start = io.read_pw_scf("files/ase/pw.scf.out.start.wz-AlN")
    at = st_start.get_ase_atoms(pbc=True)
    exe = find_exe(lammps_exe_names)
    assert exe is not None
    calc = Lammps(
        label="lmp",
        directory=tmpdir / "calc_dir",
        pair_style="tersoff",
        pair_coeff="* * AlN.tersoff Al N",
        command=f"{exe} < lmp.in > lmp.out 2>&1",
        backup=True,
    )
    at.set_calculator(calc)
    return at


# scope doesn't work with pytest-xdist ... so sharing fixtures across tests
# doesn't work. According to the pytest-xdist docs, we need to implement thread
# safety by hand using lock files :-D. All that fancy fixture stuff and now ...
# well ok. For now we give up on sharing that fixture, i.e. unpack pseudos only
# once, b/c it is really not big and/or slow, just bad style to do something
# wasteful. Instead, we rely on the pwtools_tmpdir fixture to provide
# thread-safe (i.e. unique random) dirs and simply unpack pseudos there for
# each test_pwscf_*() function.
#
# Also note that "tmpdir" here is NOT the pytest fixture "tmpdir", but a normal
# variable name, since setup_pwscf() is not a test function. And here we have
# the one big downside of pytest fixtures: Too Much Magic (added ambiguity). But
# then again, also useful introspection tools such as the "request" fixture.
#
##@pytest.fixture(scope="module")
def setup_pwscf(tmpdir):
    pseudo_dir = tmpdir / "pseudo"
    common.backtick(
        f"mkdir -pv {pseudo_dir}; "
        f"cp files/qe_pseudos/*.gz {pseudo_dir}/; "
        f"gunzip {pseudo_dir}/*.gz"
    )
    return get_atoms_with_calc_pwscf(pseudo_dir, tmpdir)


def setup_lmp(tmpdir):
    at = get_atoms_with_calc_lammps(tmpdir)
    at.rattle(stdev=0.001, seed=int(time.time()))
    common.makedirs(at.calc.directory)
    common.backtick(f"cp -v {here}/../src/pwtools/test/utils/lammps/AlN.tersoff {at.calc.directory}/")
    return at


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_pwx()")
def test_pwscf_calculator_scf(pwtools_tmpdir):
    at = setup_pwscf(pwtools_tmpdir)
    # trigger calculation here
    forces = at.get_forces()
    etot = at.get_potential_energy()
    stress = at.get_stress(voigt=False)  # 3x3

    st = io.read_pw_scf(at.calc.label + ".out")
    assert np.allclose(forces, st.forces)
    assert np.allclose(etot, st.etot)
    assert np.allclose(st.stress, -stress * constants.eV_by_Ang3_to_GPa)


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_pwx()")
def test_pwscf_calculator_vc_relax(pwtools_tmpdir):
    at = setup_pwscf(pwtools_tmpdir)
    from ase.optimize import BFGS
    from ase.constraints import UnitCellFilter
    from ase.io.trajectory import Trajectory

    traj_fn = f"{at.calc.directory}/test_pwscf_calculator_vc_relax.traj"
    os.makedirs(os.path.dirname(traj_fn), exist_ok=True)
    opt = BFGS(UnitCellFilter(at), trajectory=traj_fn)

    # LDA relaxed cell (details see comment below).
    start_cell = parse.arr2d_from_txt(
        """
        -1.97281509  0.          1.97281509
         0.          1.97281509  1.97281509
        -1.97281509  1.97281509  0."""
    )

    # This effectively checks that parsing the start struct and the conversion
    # to ASE works.
    assert np.allclose(start_cell, at.get_cell())

    # Run ASE vc-relax, but only a few steps
    opt.run(fmax=0.05)

    # This is the relaxed cell obtained by fmax=0.05 with some earlier ASE+QE
    # combo and our PBE PAW pseudos in files/qe_pseudos/.
    #
    # Since we don't pin ASE and QE versions in pwtools so far, and due to that
    # are the victim of changing defaults as well as the beneficiary of bug
    # fixes, we give up on adapting this test time and time again by changing
    # this very cell. Instead we only check if the start and end cell are the
    # same in the `at` object and the traj (an ASE test really) and else we
    # only make sure that the volume grows, which it must because
    # files/ase/pw.scf.out.start is a norm-conserving LDA struct, calculated
    # with pz-vbc.UPF, so the PBE vc-relax will make the cell a bit bigger.
    #
    ##cell = parse.arr2d_from_txt(
    ##    """
    ##    -2.01841537  0           2.01841537
    ##    0            2.01841537  2.01841537
    ##    -2.01841537  2.01841537  0"""
    ##)
    ##assert np.allclose(start_cell, at.get_cell())

    # Can't we obtain the traj directly from the `opt` object instead to going
    # thru file IO?
    ase_tr = Trajectory(traj_fn, mode="r")
    assert np.allclose(start_cell, ase_tr[0].cell)

    # optimizer in-place mods at, make sure that's in line w/ the traj
    assert np.allclose(at.cell, ase_tr[-1].cell)

    assert crys.volume_cell(ase_tr[0].cell) < crys.volume_cell(ase_tr[-1].cell)

    # at least 1 backup files must exist: pw.*.0 is the SCF run, backed up
    # in the first iter of the vc-relax
    assert os.path.exists(at.calc.infile + ".0")


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_lmp()")
def test_lammps_calculator_single_point(pwtools_tmpdir):
    at = setup_lmp(pwtools_tmpdir)
    forces = at.get_forces()
    etot = at.get_potential_energy()
    stress = at.get_stress(voigt=False)  # 3x3

    st = io.read_lammps_md_txt(at.calc.label + ".out")[0]
    assert np.allclose(forces, st.forces)
    assert np.allclose(etot, st.etot)
    assert np.allclose(
        st.stress, -stress * constants.eV_by_Ang3_to_GPa, atol=1e-10
    )


@pytest.mark.skipif("not have_ase()")
@pytest.mark.skipif("not have_lmp()")
def test_lammps_calculator_relax(pwtools_tmpdir):
    at = setup_lmp(pwtools_tmpdir)
    from ase.optimize import BFGS

    opt = BFGS(at, maxstep=0.04)
    opt.run(fmax=0.001, steps=10)
    coords_frac = parse.arr2d_from_txt(
        """
        3.3333341909920072e-01 6.6666683819841532e-01 4.4325467247779138e-03
        6.6666681184103216e-01 3.3333362368205072e-01 5.0443254824788963e-01
        3.3333341909918301e-01 6.6666683819838046e-01 3.8356759709402671e-01
        6.6666681184101539e-01 3.3333362368201563e-01 8.8356759861713752e-01
        """
    )
    assert np.allclose(coords_frac, at.get_scaled_positions(), atol=1e-2)

    # at least 1 backup files must exist
    assert os.path.exists(at.calc.infile + ".0")
    assert os.path.exists(at.calc.outfile + ".0")
    assert os.path.exists(at.calc.dumpfile + ".0")
    assert os.path.exists(at.calc.structfile + ".0")
