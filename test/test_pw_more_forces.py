# Parse verbose PWscf force printing, i.e. more then one force block per time
# step, e.g. one blok for ions, one for vdw forces, ...

import numpy as np
from pwtools.parse import PwMDOutputFile, PwSCFOutputFile
from pwtools import common
from pwtools.constants import Bohr,Ang,Ry,eV


def test_pw_more_forces():    
    fac = Ry / eV / Bohr * Ang

    # MD: london=.true.
    
    filename = 'files/pw.md_london.out'
    common.system('gunzip %s.gz' %filename)
    natoms = 141
    nstep = 10
    # traj case
    pp = PwMDOutputFile(filename=filename)
    tr = pp.get_traj()
    assert tr.natoms == natoms
    assert tr.forces.shape == (nstep,natoms,3)
    assert tr.coords.shape == (nstep,natoms,3)
    assert pp._forces_raw.shape == (nstep+1,2*natoms,3)
    assert np.allclose(tr.forces, pp._forces_raw[1:,:natoms,:] * fac)
   
    # scf case, return only 1st step
    pp = PwSCFOutputFile(filename=filename)
    st = pp.get_struct()
    assert st.natoms == natoms
    assert st.forces.shape == (natoms,3)
    assert st.coords.shape == (natoms,3)
    assert pp._forces_raw.shape == (nstep+1,2*natoms,3)
    assert np.allclose(st.forces, pp._forces_raw[0,:natoms,:] * fac)
    common.system('gzip %s' %filename)
    
    # SCF: verbosity='high' + london=.true.

    filename = 'files/pw.scf_verbose_london.out'
    common.system('gunzip %s.gz' %filename)
    
    natoms = 4
    nstep = 1

    pp = PwSCFOutputFile(filename=filename)
    st = pp.get_struct()
    assert st.natoms == natoms
    assert st.forces.shape == (natoms,3)
    assert st.coords.shape == (natoms,3)
    assert pp._forces_raw.shape == (nstep, 8*natoms,3)
    assert np.allclose(st.forces, pp._forces_raw[0,:natoms,:] * fac)
    
    common.system('gzip %s' %filename)
