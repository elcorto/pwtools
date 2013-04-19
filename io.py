# io.py
#
# struct I/O

import numpy as np
import common, crys, pwscf, parse
from pwtools.common import frepr, cpickle_load
from pwtools.constants import Ha, eV
import warnings
try:
    import h5py
except ImportError:
    warnings.warn("Cannot import h5py.") 

# Cif parser
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    warnings.warn("Cannot import CifFile from the PyCifRW package. " 
    "Parsing Cif files will not work.")

def wien_sgroup_input(struct, lat_symbol='P'):
    """Generate input for WIEN2K's ``sgroup`` symmetry analysis tool.
    
    length: can be any

    Parameters
    ----------
    struct : Structure instance
    lat_symbol : str, e.g. 'P'

    Notes
    -----
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
    empty = '\n\n'
    txt = "/ lattice type symbol\n%s" %lat_symbol
    txt += empty
    txt += "/ a b c alpha beta gamma\n"
    txt += " ".join(["%.16e"]*6) % tuple(struct.cryst_const)
    txt += empty
    txt += "/ number of atoms\n%i" %struct.natoms
    txt += empty
    txt += "/ atom list (crystal cooords)\n"
    fmt = ' '.join(['%.16e']*3)
    for sym, coord in zip(struct.symbols, struct.coords_frac):
        txt += fmt % tuple(coord) + '\n' + sym + '\n'
    return txt


def write_wien_sgroup(filename, struct, **kwds):
    """
    Write `struct` to input file for WIEN2K's ``sgroup`` symmetry analysis
    tool.

    Parameters
    ----------
    filename : str
        name of the output file
    struct : Structure
    **kwds : see wien_sgroup_input()
    """
    txt = wien_sgroup_input(struct, **kwds)
    common.file_write(filename, txt)


def write_cif(filename, struct):
    """Q'n'D Cif writer. Uses PyCifRW.
    
    length: Angstrom

    Parameters
    ----------
    filename : str
        name of output .cif file
    struct : Structure, length units Angstrom assumed        
    """
    ffmt = "%.16e"
    cf = pycifrw_CifFile.CifFile()
    block = pycifrw_CifFile.CifBlock()

    block['_cell_length_a'] = frepr(struct.cryst_const[0], ffmt=ffmt)
    block['_cell_length_b'] = frepr(struct.cryst_const[1], ffmt=ffmt)
    block['_cell_length_c'] = frepr(struct.cryst_const[2], ffmt=ffmt)
    block['_cell_angle_alpha'] = frepr(struct.cryst_const[3], ffmt=ffmt)
    block['_cell_angle_beta'] = frepr(struct.cryst_const[4], ffmt=ffmt)
    block['_cell_angle_gamma'] = frepr(struct.cryst_const[5], ffmt=ffmt)
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
    data = [struct.symbols, 
            _xyz2str(struct.coords_frac[:,0]), 
            _xyz2str(struct.coords_frac[:,1]), 
            _xyz2str(struct.coords_frac[:,2]),
            struct.symbols]
    # "loop_" with multiple columns            
    block.AddCifItem([[data_names], [data]])                
    cf['pwtools'] = block
    # maxoutlength = 2048 is default for cif 1.1 standard (which is default in
    # pycifrw 3.x). Reset default wraplength=80 b/c ASE's cif reader cannot
    # handle wrapped lines.
    common.file_write(filename, cf.WriteOut(wraplength=2048))


def write_xyz(filename, obj, name='pwtools_dummy_mol_name'):
    """Write VMD-style [VMD] XYZ file.
    
    length: Angstrom
    
    Parameters
    ----------
    filename : target file name
    obj : Trajectory or Structure
    name : str, optional
        Molecule name.

    References
    ----------
    [VMD] http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/xyzplugin.html
    """
    traj = crys.struct2traj(obj)
    xyz_str = ""
    for istep in range(traj.nstep):
        xyz_str += "%i\n%s\n%s" %(traj.natoms,
                                  name + '.%i' %(istep + 1),
                                  pwscf.atpos_str_fast(traj.symbols, 
                                                       traj.coords[istep,...]),
                                  )
    common.file_write(filename, xyz_str)


def write_axsf(filename, obj):
    """Write animated XSF file for Structure (only 1 step) or Trajectory.

    Note that forces are converted eV / Ang -> Ha / Ang.
    
    length: Angstrom
    forces: Ha / Angstrom

    Parameters
    ----------
    filename : target file name
    obj : Structure or Trajectory

    References
    ----------
    [XSF] http://www.xcrysden.org/doc/XSF.html
    """
    # Notes
    # -----
    # XSF: The XSF spec [XSF] is a little fuzzy about what PRIMCOORD actually
    #     is (fractional or cartesian Angstrom). Only the latter case results
    #     in a correctly displayed structure in xcrsyden. So we use that.
    #
    # Speed: The only time-consuming step is calling atpos_str*() in the loop
    #     b/c that transforms *every* single float to a string, which
    #     effectively is a double loop over `ccf`. No way to get faster w/ pure
    #     Python.
    #
    traj = crys.struct2traj(obj)
    # ccf = cartesian coords + forces (6 columns)
    if traj.is_set_attr('forces'):
        ccf = np.concatenate((traj.coords, traj.forces*eV/Ha), axis=-1)
    else:
        ccf = traj.coords
    axsf_str = "ANIMSTEPS %i\nCRYSTAL" %traj.nstep
    for istep in range(traj.nstep):
        axsf_str += "\nPRIMVEC %i\n%s" %(istep+1,
                                         common.str_arr(traj.cell[istep,...],
                                                        zero_eps=False))
        axsf_str += "\nPRIMCOORD %i\n%i 1\n%s" %(istep+1,
                                                 traj.natoms,
                                                 pwscf.atpos_str_fast(traj.symbols, 
                                                                      ccf[istep,...]))
    common.file_write(filename, axsf_str)


class ReadFactory(object):
    """Factory class to construct callables to parse files."""
    def __init__(self, parser=None, struct_or_traj=None):
        """
        Parameters
        ----------
        parser : one of parse.*File parsing classes
        struct_or_traj : str
            {'struct','traj'}
            Whether the callables should return parser.get_{struct,traj}()'s
            return value.
        """
        self.parser = parser
        self.struct_or_traj = struct_or_traj
    
    def __call__(self, filename, **kwds):
        """
        Parameters
        ----------
        filename : str
            Name of the file to parse.
        **kwds : keywords args
            passed to the parser class (e.g. units=...)
        """
        if self.struct_or_traj == 'struct':
            return self.parser(filename, **kwds).get_struct()
        elif self.struct_or_traj == 'traj':
            return self.parser(filename, **kwds).get_traj()
        else:
            raise StandardError("unknown struct_or_traj: %s" %struct_or_traj)

read_cif = ReadFactory(parser=parse.CifFile, 
                       struct_or_traj='struct',
                       )
read_pw_scf = ReadFactory(parser=parse.PwSCFOutputFile, 
                          struct_or_traj='struct', 
                          )
read_pw_md = ReadFactory(parser=parse.PwMDOutputFile, 
                         struct_or_traj='traj', 
                         )
read_pw_vcmd = ReadFactory(parser=parse.PwVCMDOutputFile, 
                           struct_or_traj='traj', 
                           )
read_cpmd_scf = ReadFactory(parser=parse.CpmdSCFOutputFile,
                            struct_or_traj='struct', 
                            )
read_cpmd_md = ReadFactory(parser=parse.CpmdMDOutputFile, 
                           struct_or_traj='traj', 
                           )


def write_h5(fn, dct, skip=[None]):
    """Write dictionary with arrays (or whatever HDF5 handles) to h5 file.

    Parameters
    ----------
    fn : str
        filename
    dct : dict
    skip : sequence
        Skip all ``dct[key]`` values which are in `skip`.
    """
    fh = h5py.File(fn, mode='w')
    for key,val in dct.iteritems():
        if val not in skip:
            fh[key] = val
    fh.close()


def read_h5(fn):
    """Load h5 file into dict."""
    fh = h5py.File(fn, mode='r') 
    dct = {}
    def get(name, obj, dct=dct):
        if isinstance(obj, h5py.Dataset):
            _name = name if name.startswith('/') else '/'+name
            dct[_name] = obj.value
    fh.visititems(get)            
    fh.close()
    return dct

load_h5 = read_h5
