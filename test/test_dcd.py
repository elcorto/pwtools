import numpy as np
from numpy import array, int32, int64, float32, float64
from pwtools.common import pj
from pwtools.test import tools
from pwtools import dcd, _dcd

# dcd header order from dcd.py
header_types = [\
    ('blk0-0',  'i4',1  ),  # 84 (start of first block, size=84 bytes)                    
    ('hdr',     'S4',1  ),  # 'CORD'
    ('9int',    'i4',9  ),  # mostly 0
    ('timestep','f4',1  ),  # timestep
    ('10int',   'i4',10 ),  # mostly 0, last is 24
    ('bkl0-1',  'i4',1  ),  # 84
    ('bkl1-0',  'i4',1  ),  # 164 
    ('ntitle',  'i4',1  ),  # 2 (ntitle)   
    ('remark1', 'S80',1 ),  # remark1
    ('remark2', 'S80',1 ),  # remark2
    ('blk1-1',  'i4',1  ),  # 164
    ('blk2-0',  'i4',1  ),  # 4
    ('natoms',  'i4',1  ),  # natoms
    ('blk2-1',  'i4',1  ),  # 4
    ] 

# reference headers, skip the remarks
hdr_lmp_ref = {
    'blk0-0':   array([84], dtype=int32),
    'hdr':      array(['CORD'], dtype='|S4'),
    '9int':     array([101,0,1,100,0,0,0,0,0], dtype=int32),
    'timestep': array([ 0.0005], dtype=float32),
    '10int':    array([1,0,0,0,0,0,0,0,0,24], dtype=int32),
    'bkl0-1':   array([84], dtype=int32),
    'bkl1-0':   array([164], dtype=int32),
    'ntitle':   array([2], dtype=int32),
    'blk1-1':   array([164], dtype=int32),
    'blk2-0':   array([4], dtype=int32),
    'natoms':   array([16], dtype=int32),
    'blk2-1':   array([4], dtype=int32),
    }

hdr_cp2k_ref = {
    'blk0-0':   array([84], dtype=int32),
    'hdr':      array(['CORD'], dtype='|S4'),
    '9int':     array([0,0,1,0,0,0,0,0,0], dtype=int32),
    'timestep': array([1.0], dtype=float32),
    '10int':    array([1,0,0,0,0,0,0,0,0,24], dtype=int32),
    'bkl0-1':   array([84], dtype=int32),
    'bkl1-0':   array([164], dtype=int32),
    'ntitle':   array([2], dtype=int32),
    'blk1-1':   array([164], dtype=int32),
    'blk2-0':   array([4], dtype=int32),
    'natoms':   array([57], dtype=int32),
    'blk2-1':   array([4], dtype=int32),
    }

def test_dcd():
    # nstep = 101
    # natoms = 16
    dir_lmp = tools.unpack_compressed('files/lammps/md-npt.tgz')
    fn_lmp = pj(dir_lmp, 'lmp.out.dcd') 

    # nstep = 16
    # natoms = 57
    dir_cp2k = tools.unpack_compressed('files/cp2k/dcd/npt_dcd.tgz')
    fn_cp2k = pj(dir_cp2k, 'PROJECT-pos-1.dcd')
    
    # read headers
    hdr_lmp = dcd.read_dcd_header_py(fn_lmp)
    hdr_cp2k = dcd.read_dcd_header_py(fn_cp2k)
    
    print ">>> comparing headers"
    for ref, dct in [(hdr_cp2k_ref, hdr_cp2k), (hdr_lmp_ref, hdr_lmp)]:
        tools.assert_dict_with_all_types_equal(ref, dct, keys=ref.keys(),
                                               strict=True)
    
    # compare data read by python (dcd.py) and fortran (dcd.f90, _dcd)
    # implementations
    # cc = cryst_const
    # co = coords
    print ">>> comparing data"
    for fn,convang,nstephdr,nstep,natoms in [(fn_lmp,True,True,101,16), 
                                             (fn_lmp,True,False,101,16),
                                             (fn_cp2k,False,False,16,57)]:
        cc_py, co_py = dcd.read_dcd_data_py(fn, convang=convang)
        cc_f, co_f = dcd.read_dcd_data_f(fn, convang=convang, nstephdr=nstephdr)
        # cryst_const is float64 in dcd
        print ">>> ... cryst_const"
        tools.assert_array_equal(cc_py, cc_f)
        # co_f is float32
        print ">>> ... coords"
        tools.assert_array_equal(co_py, co_f)
        print ">>> ... shapes"
        assert cc_f.shape == (nstep,6)
        assert co_f.shape == (nstep,natoms,3)
        # lmp angles are around 60, cp2k around 90 degree, cosines are between
        # -1 and 1, make sure the angle conversion works
        print ">>> ... angles"
        assert (cc_py[:,3:] > 50).all()
