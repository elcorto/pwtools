# pwscf.py
#
# Some handy tools to construct strings which for building pwscf input files.

import numpy as np
from pwtools.common import fix_eps, str_arr

def atpos_str(symbols, coords, fmt="%.16e", zero_eps=True):
    """Convenience function to make a string for the ATOMIC_POSITIONS section
    of a pw.x input file. Usually, this can be used to process the output of
    crys.scell().
    
    args:
    -----
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of coords
    coords : array (natoms, 3) with atomic coords, can also be (natoms, >3) to
        add constraints on atomic forces in PWscf
    zero_eps : bool
        Print values as 0.0 where |value| < eps

    returns:
    --------
    string

    example:
    --------
    >>> print atpos_str(['Al', 'N'], array([[0,0,0], [0,0,1.]]))
    Al      0.0000000000    0.0000000000    0.0000000000
    N       0.0000000000    0.0000000000    1.0000000000
    """
    coords = np.asarray(coords)
    assert len(symbols) == coords.shape[0], "len(symbols) != coords.shape[0]"
    _coords = fix_eps(coords) if zero_eps else coords
    txt = '\n'.join(symbols[i] + '\t' +  str_arr(row, fmt=fmt) \
        for i,row in enumerate(_coords))
    return txt        


def atspec_str(symbols, masses, pseudos):
    """Convenience function to make a string for the ATOMIC_SPECIES section
    of a pw.x input file.
    
    args:
    -----
    symbols : sequence of strings with atom symbols, (natoms,)
    masses : sequence if floats (natoms,) w/ atom masses
    pseudos : sequence of strings (natoms,) w/ pseudopotential file names

    returns:
    --------
    string

    example:
    --------
    >>>  print pwscf.atspec_str(['Al', 'N'], ['1.23', '2.34'], ['Al.UPF', 'N.UPF'])
    Al      1.23    Al.UPF
    N       2.34    N.UPF
    """
    assert len(symbols) == len(masses) == len(pseudos), \
        "len(symbols) != len(masses) != len(pseudos)"
    txt = '\n'.join(["%s\t%s\t%s" %(sym, str(mass), pp) for sym, mass, pp in
    zip(symbols, masses, pseudos)])        
    return txt      


def kpointstr(lst, base='nk'):
    """[3,3,3] -> "nk1=3,nk2=3,nk3=3" 
    
    Useful for QE's phonon toolchain ph.x, q2r.x, matdyn.x
    """
    return ','.join(['%s%i=%i' %(base, i+1, x) for i, x in enumerate(lst)])


def kpointstr_pwin(lst, shift=[0,0,0]):
    """[3,3,3] -> " 3 3 3 0 0 0" 
    
    Useful for pwscf input files.
    """
    if lst == 'gamma':
        return lst
    else:        
        return ' '.join(map(str, lst+shift))


def kpointstr_pwin2(lst, shift=[0,0,0]):
    """Full k-points string for pw.x input files.
    """
    if lst == 'gamma':
        return "K_POINTS gamma"
    else:
        return "K_POINTS automatic\n%s"  %kpointstr_pwin(lst, shift=shift)
