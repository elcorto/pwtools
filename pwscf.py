# pwscf.py
#
# Some handy tools to construct strings which for building pwscf input files.
# Readers for QE postprocessing tool output (matdyn.x  etc).

import re
import numpy as np
from pwtools.common import fix_eps, str_arr, file_readlines
from pwtools import crys
from math import sin, acos, sqrt

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


def ibrav2cell(ibrav, celldm):
    """Convert PWscf's ibrav + celldm to cell. All formulas are taken straight
    from the PWscf homepage. Don't blame me for errors. Use after testing. Ask
    you doctor.
    
    `celldm` (a = celldm[0]) is assumed be in the unit that you want for
    `cell` (Bohr, Angstrom, etc).

    Note: There are some documentation <-> forluma errors / inconsistencies for
    ibrav=12,13. See test/test_ibrav.py. If you really need that, have a look
    at the PWscf source for how they do it there.

    args:
    -----
    ibrav : int
        1 ... 14
    celldm : sequence of length 6
        This not the isame length 6 array `celldm` in crys.py. Here, the
        entries which are not needed can be None.
    
    returns:
    --------
    array (3,3) : cell vectors as rows, unit is that of celldm[0], i.e. a

    notes:
    ------
    ibrav = 14 is actually the only case where all 6 entries of `celldm` are
    needed and therefore the same as crys.cc2cell(crys.celldm2cc(celldm)).
    The returned `cell` here has the same spatial orientation as the one
    returned from crys.cc2cell(): a along x, b in xy-plane.
    """
    # some of celldm can be None
    tofloat = lambda x: x if x is None else float(x)
    aa, bb_aa, cc_aa, cos_alpha, cos_beta, cos_gamma = [tofloat(x) for x in celldm]
    if bb_aa is not None:
        bb = bb_aa * aa
    if cc_aa is not None:
        cc = cc_aa * aa
    if cos_gamma is not None:
        sin_gamma = sin(acos(cos_gamma))
    if ibrav == 1:
        # cubic P (sc), sc simple cubic
        # v1 = a(1,0,0),  v2 = a(0,1,0),  v3 = a(0,0,1)
        cell = aa * np.identity(3)
    elif ibrav == 2:
        # cubic F (fcc), fcc face centered cubic
        # v1 = a(1,0,0),  v2 = a(0,1,0),  v3 = a(0,0,1)
        cell = 0.5*aa * np.array([[-1,  0,  1], 
                                  [ 0,  1,  1], 
                                  [-1,  1,  0.0]])
    elif ibrav == 3:
        # cubic I (bcc), bcc body entered cubic
        # v1 = (a/2)(1,1,1),  v2 = (a/2)(-1,1,1),  v3 = (a/2)(-1,-1,1)
        cell = 0.5*aa * np.array([[ 1,  1,  1], 
                                  [-1,  1,  1], 
                                  [-1, -1,  1.0]])
    elif ibrav == 4:
        # Hexagonal and Trigonal P, simple hexagonal and trigonal(p)
        # v1 = a(1,0,0),  v2 = a(-1/2,sqrt(3)/2,0),  v3 = a(0,0,c/a)
        cell = aa * np.array([[ 1,    0,            0], 
                              [-0.5,  sqrt(3)/2.0,  0], 
                              [ 0,    0,            cc/aa]])
    elif ibrav == 5:
        # Trigonal R, trigonal(r)
        # v1 = a(tx,-ty,tz),   v2 = a(0,2ty,tz),   v3 = a(-tx,-ty,tz)
        #   where c=cos(alpha) is the cosine of the angle alpha between any pair
        #   of crystallographic vectors, tc, ty, tz are defined as
        # tx=sqrt((1-c)/2), ty=sqrt((1-c)/6), tz=sqrt((1+2c)
        tx = sqrt((1.0 - cos_alpha)/2.0)
        ty = sqrt((1.0 - cos_alpha)/6.0)
        tz = sqrt((1.0 + 2*cos_alpha)/3.0)
        cell = aa * np.array([[ tx,   -ty,      tz], 
                              [ 0.0,   2.0*ty,  tz], 
                              [-tx,   -ty,      tz]])
    elif ibrav == 6:
        # Tetragonal P (st), simple tetragonal (p)
        # v1 = a(1,0,0),  v2 = a(0,1,0),  v3 = a(0,0,c/a)
        cell = aa * np.array([[1,  0,  0], 
                              [0,  1,  0], 
                              [0,  0,  cc/aa]])
    elif ibrav == 7:
        # Tetragonal I (bct), body centered tetragonal (i)
        # v1 = (a/2)(1,-1,c/a),  v2 = (a/2)(1,1,c/a),  v3 = (a/2)(-1,-1,c/a)
        cell = 0.5*aa * np.array([[ 1,  -1,  cc/aa], 
                                  [ 1,   1,  cc/aa], 
                                  [-1,  -1,  cc/aa]])        
    elif ibrav == 8:
        # Orthorhombic P, simple orthorhombic (p)
        # v1 = (a,0,0),  v2 = (0,b,0), v3 = (0,0,c)
        cell = np.array([[aa,  0,  0], 
                         [0,   bb, 0], 
                         [0,   0,  cc]])
    elif ibrav == 9:
        # Orthorhombic base-centered(bco), bco base centered orthorhombic
        # v1 = (a/2,b/2,0),  v2 = (-a/2,b/2,0),  v3 = (0,0,c)
        cell = np.array([[ aa/2.0,   bb/2.0,   0], 
                         [-aa/2.0,   bb/2.0,   0], 
                         [ 0,        0,        cc]])
    elif ibrav == 10:
        # Orthorhombic face-centered, face centered orthorhombic
        # v1 = (a/2,0,c/2),  v2 = (a/2,b/2,0),  v3 = (0,b/2,c/2)
        cell = np.array([[aa/2.0,    0,       cc/2.0], 
                         [aa/2.0,    bb/2.0,  0], 
                         [0,         bb/2.0,  cc/2.0]])
    elif ibrav == 11:
        # Orthorhombic body-centered, body centered orthorhombic
        # v1 = (a/2,b/2,c/2),  v2 = (-a/2,b/2,c/2),  v3 = (-a/2,-b/2,c/2)
        cell = np.array([[ aa/2.0,   bb/2.0,  cc/2.0], 
                         [-aa/2.0,   bb/2.0,  cc/2.0],
                         [-aa/2.0,  -bb/2.0,  cc/2.0]])
    elif ibrav == 12:
        # Monoclinic P, monoclinic (p)
        # v1 = (a,0,0), v2= (b*cos(gamma), b*sin(gamma), 0),  v3 = (0, 0, c)
        cell = np.array([[aa,             0,              0], 
                         [bb*cos_gamma,  bb*sin_gamma,  0],
                         [0,              0,              cc]])
    elif ibrav == 13:
        # Monoclinic base-centered, base centered monoclinic
        # v1 = (a/2,0,-c/2), v2 = (b*cos(gamma),b*sin(gamma), 0), v3 = (a/2,0,c/2)
        cell = np.array([[aa/2.0,         0,             -cc/2.0], 
                         [bb*cos_gamma,   bb*sin_gamma,   0],
                         [aa/2.0,         0,              cc/2.0]])
    elif ibrav == 14:
        # Triclinic
        # v1 = (a, 0, 0),
        # v2 = (b*cos(gamma), b*sin(gamma), 0)
        # v3 = (c*cos(beta),  c*(cos(alpha)-cos(beta)cos(gamma))/sin(gamma),
        # c*sqrt( 1 + 2*cos(alpha)cos(beta)cos(gamma)
        #           - cos(alpha)^2-cos(beta)^2-cos(gamma)^2 )/sin(gamma) 
        v1 = np.array([aa,0,0])
        v2 = np.array([bb*cos_gamma, bb*sin_gamma, 0])
        v3 = np.array([cc*cos_beta, 
                       cc*(cos_alpha - cos_beta*cos_gamma)/sin_gamma, 
                       cc*sqrt( 1 + 2*cos_alpha*cos_beta*cos_gamma - \
                       cos_alpha**2.0-cos_beta**2.0-cos_gamma**2.0)/sin_gamma])
        cell = np.array([v1, v2, v3])
    else:
        raise StandardError("illegal ibrav: %s" %ibrav)
    return cell        
