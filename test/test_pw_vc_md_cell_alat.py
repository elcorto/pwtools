# See utils/make-vc-md-cell.py

import numpy as np
from pwtools import parse
from pwtools.test.testenv import testdir

def test_pw_vc_md_cell_alat():
    nstep = 10
    cell_single = np.arange(1,10).reshape((3,3))
    cell = np.empty((nstep,3,3))
    cell_no_unit = np.empty((nstep,3,3))
    # copy from make-vc-md-cell.py
    alat_lst = [2.0, 4.0]
    for ialat,alat in enumerate(alat_lst):
        for ii in range(5):
            this_cell = cell_single+0.02*ii + ialat
            cell[ialat*5 + ii,...] = this_cell
            cell_no_unit[ialat*5 + ii,...] = this_cell/alat

    pp = parse.PwVCMDOutputFile('files/pw.vc-md.cell.out')
    assert np.allclose(pp.get_cell(), cell)
    assert pp.get_cell_unit() == 'alat'
    assert np.allclose(pp._get_cell_step_unit(), np.array([2.0]*5 + [4.0]*5))

    pp = parse.PwMDOutputFile('files/pw.constant_cell.txt')
    assert pp._get_cell_step_unit() is None

    pp = parse.PwVCMDOutputFile('files/pw.constant_cell.txt')
    assert pp._get_cell_step_unit() is None

    # respect use_alat=False, then self.alat=1.0
    pp = parse.PwVCMDOutputFile('files/pw.vc-md.cell.out', use_alat=False)
    assert np.allclose(pp.get_cell(), cell_no_unit)
    # We found 'CELL_PARAMETERS.*alat' but don't use it.
    assert pp.get_cell_unit() == 'alat'
    assert pp._get_cell_step_unit() is None

