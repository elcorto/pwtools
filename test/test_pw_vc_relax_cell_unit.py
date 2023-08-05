# Test that we correctly parse cell_unit from
#
#   CELL_PARAMETERS (alat = 1.234)
#
# as 'alat'.

import os
from pwtools.parse import PwMDOutputFile
from pwtools import common, crys, num
from pwtools.test.tools import assert_attrs_not_none
from pwtools.test.testenv import testdir
pj = os.path.join

def test_pw_vc_relax_out():
    filename = 'files/pw.vc_relax_cell_unit.out'
    common.system('gunzip %s.gz' %filename)
    pp = PwMDOutputFile(filename=filename)
    pp.parse()
    common.system('gzip %s' %filename)
    none_attrs = ['coords',
                  'ekin',
                  'temperature',
                  'timestep',
                  ]
    assert_attrs_not_none(pp, none_attrs=none_attrs)
    traj = pp.get_traj()
    none_attrs = [\
        'ekin',
        'temperature',
        'timestep',
        'velocity',
        'time',
        ]
    assert_attrs_not_none(traj, none_attrs=none_attrs)
    assert pp.cell_unit == 'alat'
    assert pp.cell.shape == (6,3,3)
    for idx in range(1, pp.cell.shape[0]):
        assert num.rms(pp.cell[idx,...] - pp.cell[0,...]) > 0.0

    # Test _get_block_header_unit, which is used in get_cell_unit().
    dct = \
        {'FOO': None,
         'FOO alat': 'alat',
         'FOO (alat)': 'alat',
         'FOO {alat}': 'alat',
         'FOO (alat=1.23)': 'alat',
         'FOO (alat=  1.23)': 'alat',
         }

    for txt,val in dct.items():
        fn = pj(testdir, 'test_block_header_unit.txt')
        common.file_write(fn, txt)
        pp.filename = fn
        assert pp._get_block_header_unit('FOO') == val
