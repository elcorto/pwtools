# pwscf.py
#
# Some handy tools to construct strings which for building pwscf input files.
# Readers for QE postprocessing tool output (matdyn.x  etc).

import re
import numpy as np
from pwtools.common import fix_eps, str_arr, file_readlines

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


def read_matdyn_freq(filename):
    """Parse frequency file produced by matdyn.x when calculating a phonon
    dispersion on a grid (ldisp=.true., used for phonon dos) or a pre-defined
    k-path in the BZ.
    
    args:
    -----
    filename : file with k-points and phonon frequencies
    
    returns:
    --------
    kpoints : array (nks, 3)
    freqs : array (nks, nbnd)

    notes:
    ------
    `filename` has the form
        <header>
        <k-point, (3,)>
        <frequencies,(nbnd,)
        <k-point, (3,)>
        <frequencies,(nbnd,)
        ...

        example:        
        -------------------------------------------------------------
        &plot nbnd=  12, nks=  70 /
                   0.000000  0.000000  0.000000
           0.0000    0.0000    0.0000  235.7472  235.7472  452.7206
         503.6147  503.6147  528.1935  528.1935  614.2043  740.3496
                   0.000000  0.000000  0.077505
          50.9772   50.9772   74.0969  231.6412  231.6412  437.1183
         504.6069  504.6069  527.5871  527.5871  624.4091  737.7758
        ... 
        -------------------------------------------------------------
    
    see also:
    ---------
    bin/plot_dispersion.py
    """
    lines = file_readlines(filename)
    # Read number of bands (nbnd) and qpoints (nks). OK, Fortran's namelists
    # win here :)
    # nbnd: number of bands = number of frequencies per k-point = 3*natoms
    # nks: number of k-points
    pat = r'.*\s+nbnd\s*=\s*([0-9]+)\s*,\s*nks\s*=\s*([0-9]+)\s*/'
    match = re.match(pat, lines[0])
    assert (match is not None), "match is None"
    nbnd = int(match.group(1))
    nks = int(match.group(2))
    kpoints = np.empty((nks, 3), dtype=float)
    freqs = np.empty((nks, nbnd), dtype=float)
    step = 3 + nbnd
    items = np.array(' '.join(lines[1:]).split(), dtype=float)
    for ii in range(len(items) / step):
        kpoints[ii,:] = items[ii*step:(ii*step+3)]
        freqs[ii,:] = items[(ii*step+3):(ii*step+step)]
    return kpoints, freqs


