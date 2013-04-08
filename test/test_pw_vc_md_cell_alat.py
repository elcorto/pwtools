# See utils/make-vc-md-cell.py

import numpy as np
from pwtools import parse
from pwtools.test.testenv import testdir

def test_pw_vc_md_cell_alat():
    cell_single = np.arange(1,10).reshape((3,3))
    cell = np.empty((10,3,3))
    for ialat in [0,1]:
        for ii in range(5):
            cell[ialat*5 + ii,...] = cell_single+0.02*ii + ialat
    
    pp = parse.PwVCMDOutputFile('files/pw.vc-md.cell.out')
    assert np.allclose(pp.get_cell(), cell)
    assert pp.get_cell_unit() == 'alat'
    assert np.allclose(pp._get_cell_step_unit(), np.array([2.0]*5 + [4.0]*5))

    pp = parse.PwMDOutputFile('files/pw.constant_cell.txt')
    assert pp._get_cell_step_unit() is None

    pp = parse.PwVCMDOutputFile('files/pw.constant_cell.txt')
    assert pp._get_cell_step_unit() is None
