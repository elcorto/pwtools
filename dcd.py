import numpy as np

# DCD file header 
#   (name, dtype, count)
# numpy dtypes:
#   i4  = int32
#   f4  = float32 (single precision)
#   f8  = float64 (double precision)
#   S80 = string of length 80 (80 chars)
HEADER_TYPES = [\
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


def read_chunk(fd, types):
    """Read data from open binary file.

    Parameters
    ----------
    fd : open file
    types : list
        List of tuples::

            [(name1, dtype1, count1), 
             (name2, dtype2, count2),
             ...]
        
        See HEADER_TYPES for an example.
    
    Returns
    -------
    ret : dict
        Dictionary with len(`types`) arrays of `dtype`, each array has length
        `count`, read from `fd` by ``numpy.fromfile``::
        
            {name1: array([...], dtype1), 
             name2: array([...], dtype2),
             ...}
    """
    ret = dict((name, np.fromfile(fd, dtype, count)) for \
                name,dtype,count in types)
    return ret


def read_dcd_header_py(fn):
    """Shortcut function for reading the header from `fn`, using HEADER_TYPES.

    Parameters
    ----------
    fn : str
        filename
    
    Returns
    -------
    ret : dict
        see :func:`read_chunk`
    """    
    fd = open(fn, 'rb')
    ret = read_chunk(fd, HEADER_TYPES)
    fd.close()
    return ret

def read_dcd_data_py(fn, convang=False):
    """Read dcd file. Pure Python version.

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
    >>> cc,co = read_dcd_data_py('cp2k.dcd')
    >>> cc,co = read_dcd_data_py('cp2k.dcd', convang=False)
    >>> cc,co = read_dcd_data_py('lammps.dcd', convang=True)
    """
    # The list-append approach here is slow, very simple-minded and more or
    # less only a proof of concept. The Fortran version is 10x faster, even
    # though it also does something very bute force: it walkes the file twice
    # since it needs to determine nstep. Well well. One should do something
    # more clever such as allocating arrays like so:
    #   chunk_size_nstep = 100
    #   coords = np.empty((chunk_size_nstep, natoms,3))
    #   cryst_const = np.empty((chunk_size_nstep, 6))
    # and fill them in the loop and count nstep. If full, then allocate more,
    # copy old array into new, deallocate old (or concatenate arrays). At the
    # end, return a *view* which is only nstep long. Can we do this in Fortran
    # as well: allocate and grow arrays at runtime, return a view?
    fd = open(fn, 'rb')
    natoms = read_chunk(fd, HEADER_TYPES)['natoms'][0]
    data_types = [\
       ('blk0-0',           'i4',1),        # 48 = cryst_const_dcd = 6*8
       ('cryst_const_dcd',  'f8',6),        # unit cell
       ('bkl0-1',           'i4',1),        # 48
       ('blkx-0',           'i4',1),        # natoms*4
       ('x',                'f4',natoms),   # x
       ('blkx-1',           'i4',1),        # natoms*4
       ('blky-0',           'i4',1),        # natoms*4
       ('y',                'f4',natoms),   # y
       ('blky-1',           'i4',1),        # natoms*4
       ('blkz-0',           'i4',1),        # natoms*4
       ('z',                'f4',natoms),   # z
       ('blkz-1',           'i4',1),        # natoms*4
       ]
    cryst_const = []
    coords = []
    tmp_coords = np.empty((natoms,3), dtype=np.float32)
    tmp_cryst_const = np.empty((6,), dtype=np.float64)
    while True:
        data = read_chunk(fd, data_types)
        if len(data['cryst_const_dcd']) == 0:
            break
        else:
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
    # lammps
    if convang:
        cryst_const[:,3:] = np.arccos(cryst_const[:,3:])*180.0/np.pi
    return cryst_const, coords


def read_dcd_data_f(fn, convang=False, nstephdr=False):
    """Read dcd file. Wrapper for the Fortran version in ``dcd.f90``.  
    
    Parameters
    ----------
    fn : str
        filename
    convang : bool
        See :func:`read_dcd_data_py`
    nstephdr : bool
        read nstep from header (lammps) instead of walking the file twice (more
        safe but slower, works for all dcd flavors)
    
    Returns
    -------
    ret : See :func:`read_dcd_data_py`

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


##import struct, re
##rex = re.compile(r'([@=<>!|]*)([a-zA-Z])([0-9]+)')
##def fromfile_unpack(fd, dtype, count):
##    # np.fromfile, but using plain struct.unpack(), only for testing
##    ret = []
##    for i in range(count):
##        # '<f4': endi='<', typ='f', size='4'
##        endi,typ,size = rex.match(dtype).groups()
##        buf = fd.read(int(size))
##        if not buf:
##            break
##        if typ == 'S':
##            ret.append(buf)
##        else:  
##            dtype = endi + 'd8' if (typ=='f' and size=='8') else dtype
##            ret.append(struct.unpack(dtype, buf)[0])
##    return np.array(ret)

