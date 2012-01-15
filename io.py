# io.py
#
# File IO stuff (text and binary files).

from cStringIO import StringIO

import numpy as np

from pwtools.decorators import open_and_close, crys_add_doc
from pwtools.common import PydosConfigParser, frepr
from pwtools.verbose import verbose
from pwtools import common, crys, constants
from pwtools.pwscf import atpos_str, atpos_str_fast

# Cif parser
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    print("%s: Warning: Cannot import CifFile from the PyCifRW package. " 
    "Parsing Cif files will not work." %__file__)

# globals
HEADER_MAXLINES = 20
HEADER_COMMENT = '#'
TXT_MAXDIM = 3


@open_and_close
def _read_header_config(fh, header_maxlines=HEADER_MAXLINES, 
                        header_comment=HEADER_COMMENT):
    """Read a ini-style file from the header of a text file. Return a
    PydosConfigParser object.

    args:
    -----
    fh : file handle, readable
    header_maxlines : max lines to read down from top of the file
    header_comment : comment sign w/ which the header must be prefixed
    
    returns:
    --------
    PydosConfigParser object

    example:
    --------
    >>> !cat foo.txt
    # [array]
    # shape = 3
    # axis = -1
    1
    2
    3
    >>> _get_header_config('foo.txt')
    <pwtools.pydos.PydosConfigParser instance at 0x2c52320>
    """
    fn = common.get_filename(fh)
    verbose("[_read_header_config]: reading header from '%s'" %fn)
    header = ''
    for i in range(header_maxlines):
        try:
            line = fh.next().strip()
        except StopIteration:
            break
        if line.startswith(header_comment):
            header += line.replace(header_comment, '').strip() + '\n'
    # Read one more line to see if the header is bigger than header_maxlines.
    try:
        if fh.next().strip().startswith(header_comment):
            raise StandardError("header seems to be > header_maxlines (%i)"
                %header_maxlines)
    except StopIteration:
        pass
    c = PydosConfigParser()
    c.readfp(StringIO(header))
    # If header_maxlines > header size, we read beyond the header into the data. That
    # causes havoc for all functions that read fh afterwards.
    fh.seek(0)
    return c


# the open_and_close decorator cannot be used here b/c it only opens
# files in read mode, not for writing
def _write_header_config(fh, config, header_comment=HEADER_COMMENT,
                         header_maxlines=HEADER_MAXLINES):
    """Write ini-style config file from `config` prefixed with `header_comment` to
    file handle `fh`."""
    fn = common.get_filename(fh)
    verbose("[_write_header_config]: writing header to '%s'" %fn)
    # write config to dummy file
    ftmp = StringIO()
    config.write(ftmp)
    # write with comment signs to actual file
    ftmp.seek(0)
    lines = ftmp.readlines()
    common.assert_cond(len(lines) <= header_maxlines, 
                "header has more then header_maxlines (%i) lines" \
                %header_maxlines)
    for line in lines:
        fh.write(header_comment + ' ' + line)
    ftmp.close()

def _get_not_none(lst):
    """Return the one item in ``lst`` which is not None."""
    nitems = len(lst)
    assert lst.count(None) == (nitems - 1), "need %i None items" %(nitems - 1,)
    for item in lst:
        if item is not None:
            return item

def writetxt(fn, arr, axis=-1, maxdim=TXT_MAXDIM):
    """Write 1d, 2d or 3d arrays to txt file. 
    
    If 3d, write as 2d chunks. Take the 2d chunks along `axis`. Write a
    commented out ini-style header in the file with infos needed by readtxt()
    to restore the right shape.
    
    args:
    -----
    fn : filename
    arr : array (max 3d)
    axis : axis along which 2d chunks are written
    maxdim : highest number of dims that `arr` is allowed to have
    """
    verbose("[writetxt] writing: %s" %fn)
    common.assert_cond(arr.ndim <= maxdim, 'no rank > %i arrays supported' %maxdim)
    fh = open(fn, 'w+')
    c = PydosConfigParser()
    sec = 'array'
    c.add_section(sec)
    c.set(sec, 'shape', common.seq2str(arr.shape))
    c.set(sec, 'axis', axis)
    _write_header_config(fh, c)
    # 1d and 2d case
    if arr.ndim < maxdim:
        np.savetxt(fh, arr)
    # 3d        
    else:
        # TODO get rid of loop?                                                 
        # write 2d arrays, one by one
        sl = [slice(None)]*arr.ndim
        for ind in range(arr.shape[axis]):
            sl[axis] = ind
            np.savetxt(fh, arr[sl])
    fh.close()


@open_and_close
def readtxt(fh, axis=None, shape=None, header_maxlines=HEADER_MAXLINES,
            header_comment=HEADER_COMMENT, maxdim=TXT_MAXDIM, **kwargs):
    """Read arrays from .txt file using np.loadtxt(). 
    
    If the file stores a 3d array as consecutive 2d arrays (e.g. output from
    molecular dynamics code) the file header (see writetxt()) is used to
    determine the shape of the original 3d array and the array is reshaped
    accordingly.
    
    If `axis` or `shape` is not None, then these are used instead and 
    the header, if existing, will be ignored. This has the potential to shoot
    yourself in the foot. Use with care.
    
    If `axis` and `shape` are None, then this function does not work with
    normal text files which have no special header. Use np.loadtxt() in this
    case.

    args:
    -----
    fh : file_like
    axis : int
    shape : tuple
    **kwargs : keyword args passed to numpy.loadtxt(), e.g. comments='@@' to
        ignore weird lines etc.

    returns:
    --------
    nd array
    """
    fn = common.get_filename(fh)
    verbose("[readtxt] reading: %s" %fn)
    verbose("[readtxt]    axis: %s" %str(axis))
    verbose("[readtxt]    shape: %s" %str(shape))
    if shape is None or axis is None:
        c = _read_header_config(fh)
        sec = 'array'
        if shape is None:
            shape = common.str2tup(c.get(sec, 'shape'))
        if axis is None:            
            axis = int(c.get(sec, 'axis'))
    ndim = len(shape)
    common.assert_cond(ndim <= maxdim, 'no rank > %i arrays supported' %maxdim)
    # axis = -1 means the last dim
    if axis == -1:
        axis = ndim - 1
    
    # handle empty files (no data, only special header or nothing at all)
    header_lines = []
    for i in range(header_maxlines):
        try:
            line = fh.next().strip()
            if not line.startswith(header_comment) and line != '':
                header_lines.append(line)
        except StopIteration:
            break
    fh.seek(0)
    if header_lines == []:
        verbose("[readtxt] WARNING: empty file: %s" %fn)
        return np.array([])
    else:
        fh.seek(0)
        read_arr = np.loadtxt(fh, **kwargs)
    
    # 1d and 2d
    if ndim <= 2:
        arr = read_arr
    # 3d        
    else:
        # example:
        #   axis = 1
        #   shape = (50, 1000, 3)
        #   shape_2d_chunk =  (50, 3)
        shape_2d_chunk = shape[:axis] + shape[(axis+1):]
        # (50, 1000, 3) : natoms = 50, nstep = 1000, 3 = x,y,z
        arr = np.empty(shape, dtype=read_arr.dtype)
        # read_arr: (50*1000, 3)
        expect_shape = (shape_2d_chunk[0]*shape[axis],) + (shape_2d_chunk[1],)
        common.assert_cond(read_arr.shape == expect_shape, 
                    "read 2d array from '%s' has not the correct "
                    "shape, got %s, expect %s" %(fn, 
                                                 str(read_arr.shape),
                                                 str(expect_shape)))
        # TODO get rid of loop?                                                 
        sl = [slice(None)]*ndim
        for ind in range(shape[axis]):
            sl[axis] = ind
            arr[sl] = read_arr[ind*shape_2d_chunk[0]:(ind+1)*shape_2d_chunk[0], :]
    verbose("[readtxt]    returning shape: %s" %str(arr.shape))
    return arr


def readbin(fn):
    raise NotImplementedError("We use np.load() now.")


def readarr(fn, type='bin'):
    """Read bin or txt array from file `fn`."""
    verbose("[readarr] reading: %s" %fn)
    common.assert_cond(type in ['bin', 'txt'], "`type` must be 'bin' or 'txt'")
    if type == 'bin':
        return np.load(fn)
    elif type == 'txt':
        return readtxt(fn)


def writearr(fn, arr, comment=None, info=None,
             type='bin', axis=-1):
    """Write `arr` to binary (*.npy) or text file (*.txt) `fn`. Optionally,
    write a file `fn`.info with some misc information from `comment` and
    `info`. 
    
    args:
    -----
    arr : numpy ndarrray
    fn : str, filename
    comment : string
        a comment which will be written to the .info file (must start with '#')
    info : dict of dicts 
        sections for the .info file        
        example:
            {'foo': {'rofl1': 1, 'rofl2': 2},
             'bar': {'lol1': 5, 'lol2': 7}
            }
            
            will be converted to
            [foo]
            rofl1 = 1
            rofl2 = 2
            [bar]
            lol1 = 5
            lol2 = 7
    type : str, {bin,txt)
    only type == 'txt'
        axis : axis kwarg for writetxt()
    """
    common.assert_cond(type in ['bin', 'txt'], "`type` must be 'bin' or 'txt'")
    verbose("[writearr] writing: %s" %fn)
    verbose("[writearr]     shape: %s" %repr(arr.shape))
    if type == 'txt':
        verbose("[writearr]     axis: %i" %axis)
    if type == 'bin':
        np.save(fn, arr)
    else:
        writetxt(fn, arr, axis=axis)
    # .info file
    if comment is not None or info is not None:
        f = open(fn + '.info', 'w')
        if comment is not None:
            f.write(comment + '\n')
        if info is not None:
            c = PydosConfigParser()
            c = common.add_to_config(c, info) 
            c.write(f)
        f.close()


@crys_add_doc
def wien_sgroup_input(lat_symbol, symbols, atpos_crystal, cryst_const):
    """Generate input for WIEN2K's sgroup tool.

    args:
    -----
    lat_symbol : str, e.g. 'P'
    symbols : list of strings with atom symbols, (atpos_crystal.shape[0],)
    atpos_crystal : array_like (natoms, 3), crystal ("fractional") atomic
        coordinates
    %(cryst_const_doc)s

    notes:
    ------
    From sgroup's README:

    / ------------------------------------------------------------
    / in input file symbol "/" means a comment
    / and trailing characters are ignored by the program

    / empty lines are allowed

    P  /type of lattice; choices are P,F,I,C,A

    /  parameters of cell:
    /  lengths of the basis vectors and
    /  angles (degree unit is used)  alpha=b^c  beta=a^c  gamma=a^b
    /   |a|  |b|   |c|               alpha  beta  gamma

       1.0   1.1   1.2                90.   91.    92.

    /Number of atoms in the cell
    4

    /List of atoms
    0.1 0.2 0.3  / <-- Atom positions in units of the vectors a b c
    Al           / <-- name of this atom

    0.1 0.2 0.4  /....
    Al1

    0.2 0.2 0.3
    Fe

    0.1 0.3 0.3
    Fe

    / ------------------------------------------------------------------
    """
    atpos_crystal = np.asarray(atpos_crystal)
    common.assert_cond(len(symbols) == atpos_crystal.shape[0], 
        "len(symbols) != atpos_crystal.shape[0]")
    empty = '\n\n'
    txt = "/ lattice type symbol\n%s" %lat_symbol
    txt += empty
    txt += "/ a b c alpha beta gamma\n"
    txt += " ".join(["%.16e"]*6) % tuple(cryst_const)
    txt += empty
    txt += "/ number of atoms\n%i" %len(symbols)
    txt += empty
    txt += "/ atom list (crystal cooords)\n"
    fmt = ' '.join(['%.16e']*3)
    for sym, coord in zip(symbols, atpos_crystal):
        txt += fmt % tuple(coord) + '\n' + sym + '\n'
    return txt


@crys_add_doc
def write_cif(filename, coords_frac, cell, symbols):
    """Q'n'D Cif writer. Should be a method of parse.StructureFileParser ....
    stay tuned.
    
    args:
    -----
    filename : str
        name of output .cif file
    coords_frac : array (natoms,3)
        crystal coords
    %(cell_doc)s
        Unit: Angstrom, vectors are rows
    symbols : list of strings
        atom symbols
    """
    ffmt = "%.16e"
    cf = pycifrw_CifFile.CifFile()
    block = pycifrw_CifFile.CifBlock()
    symbols = list(symbols)

    # cell
    #
    # dunno why I have to convert to string here, assigning floats does not
    # work
    cryst_const = crys.cell2cc(cell)
    block['_cell_length_a'] = frepr(cryst_const[0], ffmt=ffmt)
    block['_cell_length_b'] = frepr(cryst_const[1], ffmt=ffmt)
    block['_cell_length_c'] = frepr(cryst_const[2], ffmt=ffmt)
    block['_cell_angle_alpha'] = frepr(cryst_const[3], ffmt=ffmt)
    block['_cell_angle_beta'] = frepr(cryst_const[4], ffmt=ffmt)
    block['_cell_angle_gamma'] = frepr(cryst_const[5], ffmt=ffmt)
    block['_symmetry_space_group_name_H-M'] = 'P 1'
    block['_symmetry_Int_Tables_number'] = 1
    # assigning a list produces a "loop_"
    block['_symmetry_equiv_pos_as_xyz'] = ['x,y,z']
    
    # atoms
    #
    # _atom_site_label: We just use symbols, which is then =
    #   _atom_site_type_symbol, but we *could* use that to number atoms of each
    #   specie, e.g. Si1, Si2, ..., Al1, Al2, ...
    data_names = ['_atom_site_label', 
                  '_atom_site_fract_x',
                  '_atom_site_fract_y',
                  '_atom_site_fract_z',
                  '_atom_site_type_symbol']
    _xyz2str = lambda arr: [ffmt %x for x in arr]
    data = [symbols, 
            _xyz2str(coords_frac[:,0]), 
            _xyz2str(coords_frac[:,1]), 
            _xyz2str(coords_frac[:,2]),
            symbols]
    # "loop_" with multiple columns            
    block.AddCifItem([[data_names], [data]])                
    cf['pwtools'] = block
    # maxoutlength = 2048 is default for cif 1.1 standard (which is default in
    # pycifrw 3.x). Reset default wraplength=80 b/c ASE's cif reader cannot
    # handle wrapped lines.
    common.file_write(filename, cf.WriteOut(wraplength=2048))


@crys_add_doc
def write_xyz(filename, coords_frac=None, coords_cart=None, 
              cell=None, symbols=None, name='pwtools_dummy_mol_name'):
    """Write VMD-style [VMD] XYZ file.
    
    Works for one structure (coords_*.shape = (natoms, 3)) or trajectories
    (natoms,3,nstep) with fixed cell (cell.shape = (3,3)).
    
    args:
    -----
    filename : target file name
    coords_{cart,frac} : 2d (one unit cell) or 3d array (e.g. MD trajectory)
        frac: crystal (fractional) coords,
        cart: cartesian
        2d: (natoms, 3)
        3d: (natoms, 3, nstep)
    %(cell_doc)s 
        In Angstrom units. Only needed if ``coords_frac`` given. Cell vectors
        are rows.
    symbols : list of strings (natoms,)
        atom symbols
    name : str, optional
        Molecule name.

    refs:
    -----
    [VMD] http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/xyzplugin.html
    """
    coords = _get_not_none([coords_frac, coords_cart])
    is3d = coords.ndim == 3
    atoms_axis = 0
    time_axis = -1
    natoms = coords.shape[atoms_axis]
    if is3d:
        nstep = coords.shape[time_axis]
        sl = [slice(None)]*3
        if coords_cart is None:
            coords_cart = crys.coord_trans(coords_frac, 
                                           old=cell, 
                                           new=np.identity(3),
                                           axis=1)
    else:
        nstep = 1
        sl = [slice(None)]*2
        if coords_cart is None:
            coords_cart = np.dot(coords_frac, cell)
    xyz_str = ""
    for istep in range(nstep):
        if is3d:
            sl[time_axis] = istep
        xyz_str += "%i\n%s\n%s\n" %(natoms,
                                  name + '.%i' %(istep + 1),
                                  atpos_str_fast(symbols, coords_cart[sl]),
                                  )
    common.file_write(filename, xyz_str)


def write_axsf(filename, coords_frac=None, coords_cart=None, cell=None,
               symbols=None, forces=None):
    """Write (variable-cell) animated XSF file. For only one unit cell (single
    structure), provide only 2d arrays.
    
    For fixed-cell trajectories use (i) one 2d array for `cell` or (ii) a 3d
    array where each cell[...,i] is the same.

    Note that `cell` must be in Angstrom, not the usual PWscf style scaled `cell`
    in alat units.
    
    The number of steps is determined from `coords`. So, for fixed coordinates
    but varying cell, use 3d `coords`, where each coords[...,i] is the same.

    args:
    -----
    filename : target file name
    coords_{cart,frac} : 2d (one unit cell) or 3d array (e.g. MD trajectory)
        frac: crystal (fractional) coords,
        cart: cartesian
        2d: (natoms, 3)
        3d: (natoms, 3, nstep)
    cell : 2d or 3d, shape like `coords`.
        Unit cell(s). In Angstrom units (for XSF). Basis vecs in `cell` (2d) or
        cell[...,i] (3d) are the rows.
    symbols : list of strings (natoms,)
        Atom symbols in *one* unit cell.
    forces : {None, 2d or 3d}, shape like `coords`. Optional.
        Forces on atoms in Hartree / Angstrom. If None, then forces are set to
        zero.
   
    examples:
    ---------
    >>> nstep=500; natoms=10; symbols=['H']*natoms
    
    Fixed cell + forces=0 (default).
    >>> coords=rand(natoms, 3, nstep)
    >>> cell=rand(3,3)
    
    Variable cell and coords with forces.
    >>> coords=rand(natoms, 3, nstep)
    >>> cell=rand(3,3,nstep)
    >>> forces=rand(natoms,3,nstep)

    Fixed fractional coords, varying cell: repeat `coords` nstep times.
    >>> tmp = rand(natoms,3)
    >>> coords = tmp[...,None].repeat(nstep, axis=2)
    >>> cell=rand(3,3,nstep)

    notes:
    ------
    If 2 or more 3d-arrays (trajectories) are used, then they must have the
    same number of steps. This is currently not checked.

    refs:
    -----
    [XSF] http://www.xcrysden.org/doc/XSF.html
    """
    # notes:
    # ------
    # XSF: The XSF spec [XSF] is a little fuzzy about what PRIMCOORD actually
    #     is (fractional or cartesian Angstrom). Only the latter case results
    #     in a correctly displayed structure in xcrsyden. So we use that.
    #
    # Speed: The only time-consuming step is calling atpos_str*() in the loop
    #     b/c that transforms *every* single float to a string, which
    #     effectively is a double loop over `ccf`. No way to get faster w/ pure
    #     Python.
    #
    # 3d arrays: Normally, we would calculate the coord trans coords ->
    #     coords_cart (fractional -> cartesian) before the loop using
    #     numpy.dot(). But if `cell` is 3d too (cell[...,i] = cell for each
    #     time step coords[...,i]), then there is no efficient numpy.dot trick
    #     for doing this with two 3d arrays, not even with tensordot(). At
    #     least, I didn't find one. Simple loop over nstep here. We would have
    #     to do
    #         coords[i,j,k]; i=0...natoms-1, j=0...2, k=0...nstep-1
    #         cell[l,m,n]; l,m=0..2, n=0...nstep-1, vectors are rows cell[l,:,n]
    #         coords_cart[i,m,k] = sum(j=0..2) coords[i,j,k] * cell[j,m,k]
    #     But this does not matter b/c (i) even in the loop the dot() is
    #     blazingly fast and (ii) we cannot avoid the loop anyway b/c
    #     building up the string to write has to be done in the loop.
    # 
    coords = _get_not_none([coords_frac, coords_cart])
    atoms_axis = 0
    xyz_axis = 1
    time_axis = 2
    # This causes buggy behavior. Apparently, slices are varied together!? 
    ##sl_coords, sl_cell, sl_forces = ([slice(None)]*3,)*3
    sl_coords  = [slice(None)]*3
    sl_cell  = [slice(None)]*3
    sl_forces = [slice(None)]*3
    isnstep_coords = coords.ndim == 3
    isnstep_cell = cell.ndim == 3
    _coords = coords if isnstep_coords else coords[...,None]
    _cell = cell if isnstep_cell else cell[...,None]
    nstep = _coords.shape[time_axis]
    natoms = _coords.shape[atoms_axis]
    if forces is None:
        _forces = (np.zeros((natoms,3), dtype=float))[...,None]
        isnstep_forces = False
    elif forces.ndim == 3:
        isnstep_forces = True
        _forces = forces
    else:        
        isnstep_forces = False
        _forces = forces[...,None]
    axsf_str = "ANIMSTEPS %i\nCRYSTAL" %nstep
    for istep in range(nstep):
        sl_coords[time_axis] = istep if isnstep_coords else 0
        sl_cell[time_axis] = istep if isnstep_cell else 0
        sl_forces[time_axis] = istep if isnstep_forces else 0
        # ccf = cartesian coords + forces for this step
        ccf = np.empty((natoms,6), dtype=float)
        if coords_cart is None:
            ccf[:,:3] = np.dot(_coords[sl_coords], _cell[sl_cell])
        else:
            ccf[:,:3] = _coords[sl_coords]
        ccf[:,3:] = _forces[sl_forces]
        # for now PRIMVEC == CONVVEC
        axsf_str += "\nPRIMVEC %i\n%s\nCONVVEC %i\n%s" %(istep+1,
                                                         common.str_arr(_cell[sl_cell],
                                                                        zero_eps=False),
                                                         istep+1,                  
                                                         common.str_arr(_cell[sl_cell],
                                                                        zero_eps=False))
        axsf_str += "\nPRIMCOORD %i\n%i 1\n%s" %(istep+1,
                                                 natoms,
                                                 atpos_str_fast(symbols, 
                                                                ccf))
    common.file_write(filename, axsf_str)
