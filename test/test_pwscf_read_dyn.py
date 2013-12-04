import numpy as np
from pwtools import pwscf, parse
from pwtools.test.tools import aae, aaae

# matdyn.freq
kpoints_ref = \
    np.array([[0.1, 0.2, 0.3],
              [0.2, 0.4, 0.6]])

freqs_ref = \
    np.array([[1,2,3,4,5,6.1,],
              [7, 8, 9e-7,10, 11, 12]])

# matdyn.modes and ph.dyn{1,2}
qpoints_ref = \
    np.array([[ 0. ,  0. ,  0. ],
              [ 0. ,  0. ,  0.5]])

freqs_modes_ref = \
    np.array([[  0.,   7.,  14.,  21.,  28.,  35.],
              [ 42.,  49.,  56.,  63.,  70.,  77.]])

vecs_real_ref = \
    np.array([[[[  1.,   2.,   3.],
                [  4.,   5.,   6.]],
               [[  8.,   9.,  10.],
                [ 11.,  12.,  13.]],
               [[ 15.,  16.,  17.],
                [ 18.,  19.,  20.]],
               [[ 22.,  23.,  24.],
                [ 25.,  26.,  27.]],
               [[ 29.,  30.,  31.],
                [ 32.,  33.,  34.]],
               [[ 36.,  37.,  38.],
                [ 39.,  40.,  41.]]],
              [[[ 43.,  44.,  45.],
                [ 46.,  47.,  48.]],
               [[ 50.,  51.,  52.],
                [ 53.,  54.,  55.]],
               [[ 57.,  58.,  59.],
                [ 60.,  61.,  62.]],
               [[ 64.,  65.,  66.],
                [ 67.,  68.,  69.]],
               [[ 71.,  72.,  73.],
                [ 74.,  75.,  76.]],
               [[ 78.,  79.,  80.],
                [ 81.,  82.,  83.]]]])

# See test/utils/gen_matdyn_modes.py. ph.dyn{1,2} copied and adapted by hand
# from generated matdyn.modes .
vecs_imag_ref = 0.03*vecs_real_ref

def test_read_matdyn():
    # matdyn.freq
    kpoints, freqs = pwscf.read_matdyn_freq('files/matdyn.freq')
    
    aae(kpoints, kpoints_ref)
    aae(freqs, freqs_ref)
    
    # matdyn modes: read_matdyn_modes()
    qpoints, freqs, vecs = pwscf.read_matdyn_modes('files/matdyn.modes',
                                                   natoms=2)
    aae(qpoints, qpoints_ref)
    aae(freqs, freqs_modes_ref)
    aae(vecs.real, vecs_real_ref)
    aaae(vecs.imag, vecs_imag_ref)

def test_read_all_dyn():
    # matdyn modes: read_all_dyn()
    qpoints, freqs, vecs = pwscf.read_all_dyn('files/dyn/', nqpoints=2, 
                                              natoms=2, base='ph.dyn')
    aae(qpoints, qpoints_ref)
    aae(freqs, freqs_modes_ref)
    aae(vecs.real, vecs_real_ref)
    aaae(vecs.imag, vecs_imag_ref)


def test_read_dynmat():
    table_txt = """
# mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1      0.00    0.0000    0.0000         0.0005    0.7414
    2      0.00    0.0000    0.0000         0.0005    0.7465
    3      0.00    0.0000    0.0000         0.0018    0.2647
    4    252.27    7.5627    0.0000         0.0073    0.7500
    5    252.27    7.5627    0.0000         0.0073    0.7500
    6    548.44   16.4419    0.0000         0.0000    0.7434
    7    603.32   18.0872   35.9045        18.9075    0.7366
    8    656.82   19.6910    0.0000         7.9317    0.7500
    9    656.82   19.6910    0.0000         7.9317    0.7500
   10    669.67   20.0762   31.5712         5.0265    0.7500
   11    738.22   22.1311    0.0000         0.0000    0.7306
   12    922.64   27.6600   31.5712         5.0265    0.7500
"""

    vecs_txt = """
   0.03895  -0.03122  -0.00290
   0.03895  -0.03122  -0.00290
   0.03895  -0.03122  -0.00290
   0.03895  -0.03122  -0.00290
  -0.03116  -0.03906   0.00186
  -0.03116  -0.03906   0.00186
  -0.03116  -0.03906   0.00186
  -0.03116  -0.03906   0.00186
   0.00343  -0.00036   0.04988
   0.00343  -0.00036   0.04988
   0.00343  -0.00036   0.04988
   0.00343  -0.00036   0.04988
   0.02589  -0.04674   0.00000
  -0.02589   0.04674   0.00000
  -0.02244   0.04051   0.00000
   0.02244  -0.04051   0.00000
   0.04674   0.02589   0.00000
  -0.04674  -0.02589   0.00000
  -0.04051  -0.02244   0.00000
   0.04051   0.02244   0.00000
   0.00000   0.00000   0.07029
   0.00000   0.00000  -0.07029
   0.00000   0.00000  -0.00766
   0.00000   0.00000   0.00766
   0.00000   0.00000  -0.03258
   0.00000   0.00000  -0.03258
   0.00000   0.00000   0.06276
   0.00000   0.00000   0.06276
  -0.02867  -0.00445   0.00000
   0.02867   0.00445   0.00000
  -0.06372  -0.00990   0.00000
   0.06372   0.00990   0.00000
   0.00445  -0.02867   0.00000
  -0.00445   0.02867   0.00000
   0.00990  -0.06372   0.00000
  -0.00990   0.06372   0.00000
   0.00000   0.03258   0.00000
   0.00000   0.03258   0.00000
   0.00000  -0.06276   0.00000
   0.00000  -0.06276   0.00000
   0.00000   0.00000  -0.00399
   0.00000   0.00000   0.00399
   0.00000   0.00000  -0.07060
   0.00000   0.00000   0.07060
  -0.03258   0.00000   0.00000
  -0.03258   0.00000   0.00000
   0.06276   0.00000   0.00000
   0.06276   0.00000   0.00000
"""
    arr = parse.arr2d_from_txt(table_txt)
    freqs_ref = arr[:,1]
    ir_ref = arr[:,3]
    raman_ref = arr[:,4]
    depol_ref = arr[:,5]
    vecs_flat = parse.arr2d_from_txt(vecs_txt)
    natoms = 4
    nmodes = 3*natoms
    vecs_ref = np.empty((nmodes, natoms, 3), dtype=float)
    for imode in range(nmodes):
        for iatom in range(natoms):
            idx = imode*natoms + iatom
            vecs_ref[imode, iatom,:] = vecs_flat[idx, :]

    qpoints,freqs,vecs = pwscf.read_dynmat(path='files/dynmat',
                                           filename='dynmat_all.out',
                                           axsf='dynmat.axsf',
                                           natoms=natoms)
    assert np.allclose(freqs_ref, freqs)
    assert np.allclose(vecs_ref, vecs)
    assert (qpoints == np.array([0,0,0])).all()

    dct = pwscf.read_dynmat_ir_raman(filename='files/dynmat/dynmat_all.out',
                                     natoms=natoms)
    assert np.allclose(dct['freqs'], freqs_ref)
    assert np.allclose(dct['ir'], ir_ref)
    assert np.allclose(dct['raman'], raman_ref)
    assert np.allclose(dct['depol'], depol_ref)
    
    dct = pwscf.read_dynmat_ir_raman(filename='files/dynmat/dynmat_min.out',
                                     natoms=natoms)
    assert np.allclose(dct['freqs'], freqs_ref)
    assert np.allclose(dct['ir'], ir_ref)
    assert dct['raman'] is None
    assert dct['depol'] is None

