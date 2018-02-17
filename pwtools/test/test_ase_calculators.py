import os, time
import numpy as np
from pwtools import io, constants, common, parse
from pwtools.calculators import Pwscf, Lammps
from pwtools.test.tools import skip
from .testenv import testdir
pj = os.path.join

prefix = 'ase_calculator'

def have_ase():
    try:
        import ase
        return True
    except ImportError:
        return False

def have_pwx():
    for path in os.environ['PATH'].split(':'):
        if os.path.exists("%s/pw.x" %path):
            return True
    return False            

def have_lmp():
    for path in os.environ['PATH'].split(':'):
        if os.path.exists("%s/lammps" %path):
            return True
    return False            
    
def get_atoms_with_calc_pwscf(pseudo_dir):
    st_start = io.read_pw_scf('files/ase/pw.scf.out.start.rs-AlN')
    at = st_start.get_ase_atoms(pbc=True)
    label = pj(testdir, prefix, 'calc_dir', 'pw')
    calc = Pwscf(label=label,
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
    label = pj(testdir, prefix, 'calc_dir', 'lmp')
    calc = Lammps(label=label,
                  pair_style='tersoff',
                  pair_coeff='* * AlN.tersoff Al N',
                  command='lammps < lmp.in > lmp.out 2>&1',
                  backup=True,
                  ) 
    at.set_calculator(calc)
    return at        

def test_pwscf_calculator():
    if not have_ase():
        skip("no ASE found, skipping test")
    elif not have_pwx():
        skip("no pw.x found, skipping test")
    else:
        pseudo_dir = pj(testdir, prefix, 'pseudo')
        print(common.backtick("mkdir -pv {p}; cp files/qe_pseudos/*.gz {p}/; \
            gunzip {p}/*".format(p=pseudo_dir)))
        at = get_atoms_with_calc_pwscf(pseudo_dir)

        print("scf")
        # trigger calculation here
        forces = at.get_forces()
        etot = at.get_potential_energy()
        stress = at.get_stress(voigt=False) # 3x3
        
        st = io.read_pw_scf(at.calc.label + '.out')
        assert np.allclose(forces, st.forces)
        assert np.allclose(etot, st.etot)
        assert np.allclose(st.stress, -stress * constants.eV_by_Ang3_to_GPa)
        
        # files/ase/pw.scf.out.start is a norm-conserving LDA struct,
        # calculated with pz-vbc.UPF, so the PBE vc-relax will make the cell
        # a bit bigger
        print("vc-relax")
        from ase.optimize import BFGS
        from ase.constraints import UnitCellFilter
        opt = BFGS(UnitCellFilter(at))
        cell = parse.arr2d_from_txt("""
            -1.97281509  0.          1.97281509
             0.          1.97281509  1.97281509
            -1.97281509  1.97281509  0.""")        
        assert np.allclose(cell, at.get_cell())
        opt.run(fmax=0.05) # run only 2 steps
        cell = parse.arr2d_from_txt("""
            -2.01837531  0.          2.01837531
             0.          2.01837531  2.01837531
            -2.01837531  2.01837531  0""")
        assert np.allclose(cell, at.get_cell())

        # at least 1 backup files must exist: pw.*.0 is the SCF run, backed up
        # in the first iter of the vc-relax
        assert os.path.exists(at.calc.infile + '.0')


def test_lammps_calculator():
    if not have_ase():
        skip("no ASE found, skipping test")
    elif not have_lmp():
        skip("no lammps found, skipping test")
    else:
        at = get_atoms_with_calc_lammps()
        at.rattle(stdev=0.001, seed=int(time.time()))
        common.makedirs(at.calc.directory)
        print(common.backtick("cp -v utils/lammps/AlN.tersoff {p}/".format(
            p=at.calc.directory)))

        print("scf")
        forces = at.get_forces()
        etot = at.get_potential_energy()
        stress = at.get_stress(voigt=False) # 3x3
        
        st = io.read_lammps_md_txt(at.calc.label + '.out')[0]
        assert np.allclose(forces, st.forces)
        assert np.allclose(etot, st.etot)
        assert np.allclose(st.stress, -stress * constants.eV_by_Ang3_to_GPa,
                           atol=1e-10)
        
        print("relax")
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
