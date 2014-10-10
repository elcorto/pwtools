import os, warnings
import numpy as np
from pwtools import io, constants, common, parse
from pwtools.calculators import PwtoolsQE
from testenv import testdir
pj = os.path.join

prefix = 'ase_qe_calculator'

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
    
def get_atoms_with_calc(pseudo_dir):
    st_start = io.read_pw_scf('files/ase/pw.scf.out.start')
    at = st_start.get_ase_atoms(pbc=True)
    label = pj(testdir, prefix, 'calc_dir', 'pw')
    calc = PwtoolsQE(label=label,
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

def test_calculator():
    if not have_ase():
        warnings.warn("no ASE found, skipping test")
    elif not have_pwx():
        warnings.warn("no pw.x found, skipping test")
    else:
        pseudo_dir = pj(testdir, prefix, 'pseudo')
        print common.backtick("mkdir -pv {p}; cp files/qe_pseudos/*.gz {p}/; \
            gunzip {p}/*".format(p=pseudo_dir))
        at = get_atoms_with_calc(pseudo_dir)

        print "scf"
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
        print "vc-relax"
        from ase.optimize import BFGS
        from ase.constraints import UnitCellFilter
        ucf = UnitCellFilter(at)
        opt = BFGS(ucf)
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
        assert os.path.exists(at.calc.pwin + '.0')
