# coding: utf8

"""Read dcd files. Some timings (in ipython). Reading lammps files, not using
`convang` here, so angles are not converted but this is only a speed test.

::

    >>> %timeit cc,co=dcd.read_dcd_data_ref('lmp.out.dcd')
    100 loops, best of 3: 3.35 ms per loop

    >>> %timeit cc,co=dcd.read_dcd_data_f('lmp.out.dcd')
    1000 loops, best of 3: 455 µs per loop

    >>> %timeit cc,co=dcd.read_dcd_data_f('lmp.out.dcd', nstephdr=True)
    1000 loops, best of 3: 241 µs per loop

    >>> # pure numpy wins!
    >>> %timeit cc,co=dcd.read_dcd_data('lmp.out.dcd')
    10000 loops, best of 3: 114 µs per loop
"""

import numpy as np
import os


# DCD file header. Define structured dtype with field specs
#   (name, dtype, shape)
# numpy dtypes:
#   i4  = int32
#   f4  = float32 (single precision)
#   f8  = float64 (double precision)
#   S80 = string of length 80 (80 chars)
# shape:
#   old syntax:
#       ('foo', 'i4', 1) -> scalar int32
#   new syntax: don't use just a blank "1"
#       ('foo', 'i4', (1,)) -> array([int32])
#       ('foo', 'i4')       -> scalar int32
#   for array fields, both work:
#       ('foo', 'i4',    5) -> array([int32, int32, ...])
#       ('foo', 'i4', (5,)) -> array([int32, int32, ...])
#   but to be contistent better use tuple syntax
HEADER_TYPES = [
    ('blk0-0',  'i4'       ),  # 84 (start of first block, size=84 bytes)
    ('hdr',     'S4'       ),  # 'CORD'
    ('9int',    'i4',  (9,)),  # 9 ints, mostly 0
    ('timestep','f4'       ),  # timestep (float32)
    ('10int',   'i4', (10,)),  # 10 ints, mostly 0, last is 24
    ('blk0-1',  'i4'       ),  # 84
    ('blk1-0',  'i4'       ),  # 164
    ('ntitle',  'i4'       ),  # 2
    ('remark1', 'S80'      ),  # remark1
    ('remark2', 'S80'      ),  # remark2
    ('blk1-1',  'i4'       ),  # 164
    ('blk2-0',  'i4'       ),  # 4 (4 bytes = int32)
    ('natoms',  'i4'       ),  # natoms (int32)
    ('blk2-1',  'i4'       ),  # 4
    ]

HEADER_DTYPE = np.dtype(HEADER_TYPES)


def read_dcd_header(fn):
    """Shortcut function for reading the header from `fn`, using HEADER_DTYPE.

    Parameters
    ----------
    fn : str
        filename

    Returns
    -------
    ret : dict
    """
    fd = open(fn, 'rb')
    arr = np.fromfile(fd, HEADER_DTYPE, 1)
    fd.close()
    return dict((key,arr[0][key]) for key in arr.dtype.names)


def read_dcd_data_ref(fn, convang=False):
    """Read dcd file. Pure Python version. Slow, only reference implementation.
    Use :func:`read_dcd_data` instead.

    Parameters
    ----------
    fn : str
        filename
    convang : bool
        convert angles from cosine to degree (only useful for lammps style dcd
        files)

    Returns
    -------
    ret : (cryst_const, coords)
        | cryst_const : (nstep,6) float64 array, (a,b,c,alpha,beta,gamma),
        |               Angstrom, degrees
        | coords : (nstep, natoms, 3) float32 array, cartesian coords Angstrom

    Examples
    --------
    >>> # default settings read cp2k files
    >>> cc,co = read_dcd_data_ref('cp2k.dcd')
    >>> cc,co = read_dcd_data_ref('cp2k.dcd', convang=False)
    >>> cc,co = read_dcd_data_ref('lammps.dcd', convang=True)
    """
    fd = open(fn, 'rb')
    natoms = np.fromfile(fd, HEADER_DTYPE, 1)[0]['natoms']
    # data per timestep
    data_dtype = np.dtype([
       ('blk0-0',          'i4'           ),   # 48 = 6*8 bytes = 6*float64
       ('cryst_const_dcd', 'f8',      (6,)),   # unit cell (6*float64)
       ('bkl0-1',          'i4'           ),   # 48
       ('blkx-0',          'i4'           ),   # natoms*4 = natoms*float32
       ('x',               'f4', (natoms,)),   # x (natoms*float32)
       ('blkx-1',          'i4'           ),   # natoms*4
       ('blky-0',          'i4'           ),   # natoms*4
       ('y',               'f4', (natoms,)),   # y
       ('blky-1',          'i4'           ),   # natoms*4
       ('blkz-0',          'i4'           ),   # natoms*4
       ('z',               'f4', (natoms,)),   # z
       ('blkz-1',          'i4'           ),   # natoms*4
       ])
    cryst_const = []
    coords = []
    tmp_coords = np.empty((natoms,3), dtype=np.float32)
    tmp_cryst_const = np.empty((6,), dtype=np.float64)
    while True:
        _data = np.fromfile(fd, data_dtype, 1)
        if len(_data) == 0:
            break
        else:
            data = _data[0]
            tmp_coords[:,0] = data['x']
            tmp_coords[:,1] = data['y']
            tmp_coords[:,2] = data['z']
            tmp_cryst_const[0] = data['cryst_const_dcd'][0]
            tmp_cryst_const[1] = data['cryst_const_dcd'][2]
            tmp_cryst_const[2] = data['cryst_const_dcd'][5]
            tmp_cryst_const[3] = data['cryst_const_dcd'][4]
            tmp_cryst_const[4] = data['cryst_const_dcd'][3]
            tmp_cryst_const[5] = data['cryst_const_dcd'][1]
            coords.append(tmp_coords.copy())
            cryst_const.append(tmp_cryst_const.copy())
    fd.close()
    coords = np.array(coords, dtype=np.float32)
    cryst_const = np.array(cryst_const, dtype=np.float64)
    if convang:
        cryst_const[:,3:] = np.arccos(cryst_const[:,3:])*180.0/np.pi
    return cryst_const, coords


def read_dcd_data(fn, convang=False):
    """Read dcd file. Fastest version. Calculates nstep from bytes between
    end-of-header and EOF.

    Parameters
    ----------
    fn : str
        filename
    convang : bool
        convert angles from cosine to degree (only useful for lammps style dcd
        files)

    Returns
    -------
    ret : (cryst_const, coords)
        | cryst_const : (nstep,6) float64 array, (a,b,c,alpha,beta,gamma),
        |               Angstrom, degrees
        | coords : (nstep, natoms, 3) float32 array, cartesian coords Angstrom

    Examples
    --------
    >>> # default settings read cp2k files
    >>> cc,co = read_dcd_data('cp2k.dcd')
    >>> cc,co = read_dcd_data('cp2k.dcd', convang=False)
    >>> cc,co = read_dcd_data('lammps.dcd', convang=True)
    """
    fd = open(fn, 'rb')
    natoms = np.fromfile(fd, HEADER_DTYPE, 1)[0]['natoms']
    fd_pos = fd.tell()
    # seek to end
    fd.seek(0, os.SEEK_END)
    # number of bytes between fd_pos and end
    fd_rest = fd.tell() - fd_pos
    # reset to pos after header
    fd.seek(fd_pos)
    # calculate nstep: fd_rest / bytes_per_timestep
    # 4 - initial 48
    # 6*8 - cryst_const_dcd
    # 7*4 - markers between x,y,z and at the end of the block
    # 3*4*natoms - float32 cartesian coords
    nstep = fd_rest / (4 + 6*8 + 7*4 + 3*4*natoms*1.0)
    assert nstep % 1.0 == 0.0, ("calculated nstep is not int, cannot "
                                "read file '{}'".format(fn))
    nstep = int(nstep)
    # dtype for fromfile: nstep times dtype of a timestep data block
    dtype = \
        np.dtype(([('x0', 'i4'),
                   ('x1', 'f8', (6,)),
                   ('x2', 'i4', (2,)),
                   ('x3', 'f4', (natoms,)),
                   ('x4', 'i4', (2,)),
                   ('x5', 'f4', (natoms,)),
                   ('x6', 'i4', (2,)),
                   ('x7', 'f4', (natoms,)),
                   ('x8', 'i4')],
                   (nstep,)))
    arr = np.fromfile(fd, dtype, 1)
    fd.close()
    cryst_const = np.empty((nstep,6), dtype=np.float64)
    cryst_const[:,0] = arr['x1'][0,:,0]
    cryst_const[:,1] = arr['x1'][0,:,2]
    cryst_const[:,2] = arr['x1'][0,:,5]
    cryst_const[:,3] = arr['x1'][0,:,4]
    cryst_const[:,4] = arr['x1'][0,:,3]
    cryst_const[:,5] = arr['x1'][0,:,1]
    coords = np.empty((nstep,natoms,3), dtype=np.float32)
    coords[...,0] = arr['x3'][0,...]
    coords[...,1] = arr['x5'][0,...]
    coords[...,2] = arr['x7'][0,...]
    if convang:
        cryst_const[:,3:] = np.arccos(cryst_const[:,3:])*180.0/np.pi
    return cryst_const, coords


def read_dcd_data_f(fn, convang=False, nstephdr=False):
    """Read dcd file. Wrapper for the Fortran version in ``dcd.f90``.
    Deprecated, use :func:`read_dcd_data` instead.

    Parameters
    ----------
    fn : str
        filename
    convang : bool
        See :func:`read_dcd_data`
    nstephdr : bool
        read nstep from header (lammps) instead of walking the file twice (more
        safe but slower, works for all dcd flavors)

    Returns
    -------
    ret : See :func:`read_dcd_data`

    Examples
    --------
    >>> # default settings read cp2k files
    >>> cc,co = read_dcd_data_f('cp2k.dcd')
    >>> cc,co = read_dcd_data_f('cp2k.dcd', convang=False, nstephdr=False)
    >>> cc,co = read_dcd_data_f('lammps.dcd', convang=True, nstephdr=True)
    >>> # more safe if you don't trust nstep from the header
    >>> cc,co = read_dcd_data_f('lammps.dcd', convang=True, nstephdr=False)
    """
    from pwtools import _dcd
    nstep, natoms, timestep = _dcd.get_dcd_file_info(fn, nstephdr)
    return _dcd.read_dcd_data(fn, nstep, natoms, convang)
