# Test parsing the correct SCF cell from a [vc-]relax run (QE 5.x only, IIRC),
# which performs a final SCF run after the relax has converged.

import os
import numpy as np
from pwtools import common, crys, parse, io

# from pw.out: "reduced cell" = cell / alat
cell_2d_red_ref = parse.arr2d_from_txt("""
    1.000000   0.000000   0.000000  
    0.000000   2.902399   0.000000  
    0.000000   0.000000   2.304846  
    """)

def test_scf_cell():
    filename = 'files/pw.vc-relax_coords_fixed.out'
    common.system('gunzip %s.gz' %filename)
    
    pp = parse.PwSCFOutputFile(filename, use_alat=False)
    cell_2d_red = pp.get_cell()
    assert np.allclose(cell_2d_red, cell_2d_red_ref, atol=1e-15, rtol=0)
    
    pp = parse.PwSCFOutputFile(filename, use_alat=True)
    assert np.allclose(pp.get_cell(), 
                       cell_2d_red*pp.get_alat(), 
                       atol=1e-15,
                       rtol=0)
    
    st = io.read_pw_scf(filename)
    tr = io.read_pw_md(filename)

    # tr is from a vc-relax w/ fixed fractional coords, check that
    assert np.allclose(np.zeros((tr.nstep,tr.natoms,3)), 
                       tr.coords_frac - tr.coords_frac[0,...].copy(),
                       rtol=0, atol=1e-15)

    # check if scf parser gets the same coords_frac as the trajectory parser
    # Note: this and the next test have the same max error of
    # 4.33868466709e-08 (b/c of limited accuracy in printed numbers in
    # pwscf output)
    assert np.allclose(st.coords_frac,tr.coords_frac[0,...], atol=1e-7, rtol=0)
    
    # same test, plus test of concatenate() works
    trcat = crys.concatenate((st,tr))
    assert np.allclose(np.zeros((trcat.nstep,trcat.natoms,3)), 
                       trcat.coords_frac - trcat.coords_frac[0,...].copy(),
                       rtol=0, atol=1e-7)
   
