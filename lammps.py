import time
import numpy as np
from pwtools import atomic_data, common


def struct_str(struct):
    """Convert Structure object to lammps format. 
    
    The returned string can be written to a file and read in a lammps input
    file by ``read_data``.
    
    Parameters
    ----------
    struct : Structure
    
    Returns
    -------
    str : string

    References
    ----------
    ase.calculators.lammpsrun (ASE 3.8).
    """
    # align cell to [[x,0,0],[xy,y,0],[xz, yz, z]] (row format, the transpose
    # is what lammps uses)
    st = struct.copy()
    st.coords = None
    st.cell = None
    st.set_all()
    head_str = "structure written by pwtools {}".format(time.asctime())        
    info_str = '%i atoms\n%i atom types' %(st.natoms, len(st.symbols_unique))
    cell_str = "0.0 {x:.14g} xlo xhi\n0.0 {y:.14g} ylo yhi\n0.0 {z:.14g} zlo zhi\n"
    cell_str += "{tilts} xy xz yz\n"
    cell_str = cell_str.format(x=st.cell[0,0], 
                               y=st.cell[1,1], 
                               z=st.cell[2,2],
                               tilts=common.str_arr(np.array([st.cell[1,0],
                                                              st.cell[2,0],
                                                              st.cell[2,1]]),
                                                    eps=1e-13, fmt='%.14g',
                                                    delim=' '))
    atoms_str = "Atoms\n\n"
    for iatom in range(st.natoms):
        atoms_str += "{iatom} {ispec} {xyz}".format(
            iatom=iatom+1, 
            ispec=st.order[st.symbols[iatom]],
            xyz=common.str_arr(st.coords[iatom,:], eps=1e-13, fmt='%23.16e') + '\n')
    mass_str = "Masses\n\n"
    for idx,sy in enumerate(st.symbols_unique):
        mass_str += "%i %g\n" %(idx+1, atomic_data.pt[sy]['mass'])
    return head_str + '\n\n' + info_str + '\n\n' + cell_str + \
           '\n' + atoms_str + '\n' + mass_str
