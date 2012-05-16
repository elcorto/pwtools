# pwscf.py
#
# Some handy tools to construct strings for building pwscf input files.
# Readers for QE postprocessing tool output (matdyn.x  etc).

import re
import numpy as np
from pwtools.common import fix_eps, str_arr, file_readlines
import crys
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
    txt = '\n'.join("%s\t%s" %(symbols[i], str_arr(row, fmt=fmt, zero_eps=False)) \
        for i,row in enumerate(_coords))
    return txt        


def atpos_str_fast(symbols, coords, work=None):
    """Fast version of atpos_str() for usage in loops. We use a fixed string
    dtype '|S20' to convert the array `coords` to string form. We also avoid
    all assert's etc for speed.
    
    args:
    -----
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of coords
    coords : array (natoms, 3) with atomic coords, can also be (natoms, >3) to
        add constraints on atomic forces in PWscf
    work : optional, array of shape (coords.shape[0], coords.shape[1]+1)
        Pre-allocated work array. This can be the result of numpy.empty(). It
        is used to temporarily store the string dtype array.

    returns:
    --------
    string
    """
    # The string dtype + flatten trick is the fastest way to convert a numpy
    # array to string. However the number of digits is limited to 11 (at least
    # on my 64 bit machine). Formatting like '%e' cannot be used. It is like a
    # fixed digit form of '%f'. Needs about 2/3 of the time of atpos_str(), so
    # the speedup is OK, but not very high. String operations are slow. The
    # next thing would be Cython or so.
    # work: Even allocations in a loop are fast, so using `work` brings next to
    # nothing.
    nrows = coords.shape[0]
    ncols = coords.shape[1]
    arr = np.empty((nrows, ncols+1), dtype='|S20') if work is None else work
    arr[:,0] = symbols
    arr[:,1:] = coords
    txt = ('  '.join(['%s']*(ncols+1)) + '\n')*nrows %tuple(arr.flatten())
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
    """Parse frequency file produced by QE's matdyn.x ("flfrq" in matdyn.x
    input, usually "matdyn.freq" or so) when calculating a phonon dispersion on
    a grid (ldisp=.true., used for phonon dos)  or a pre-defined k-path in the
    BZ.
    
    args:
    -----
    filename : file with k-points and phonon frequencies
    
    returns:
    --------
    kpoints : array (nks, 3)
        Array with `nks` k-points.
    freqs : array (nks, nbnd)
        Array with `nbnd` energies/frequncies at each of the `nks` k-points.
        For phonon DOS, nks == 3*natoms.

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
    bin/plot_dispersion.py, kpath.py
    """
    lines = file_readlines(filename)
    # Read number of bands (nbnd) and k-points (nks). OK, Fortran's namelists
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
    # nasty trick: join all lines containing data into one 1d array: " ".join()
    # does "1 2\n3 4" -> "1 2\n 3 4" and split() splits at \n + whitespace.
    items = np.array(' '.join(lines[1:]).split(), dtype=float)
    for ii in range(len(items) / step):
        kpoints[ii,:] = items[ii*step:(ii*step+3)]
        freqs[ii,:] = items[(ii*step+3):(ii*step+step)]
    return kpoints, freqs

# XXX do we really need this function? 
# XXX the only PWscf-specific thing is that we use read_matdyn_freq() inside.
def parse_dis(fn_freq, fn_kpath_def=None):
    """Parse frequency file produced by matdyn.x (flfrq, see
    read_matdyn_freq()) and, optionally a k-path definition file.
    
    This is a helper for bin/plot_dispersion.py. It lives here b/c it is
    PWscf-specific.

    args:
    ----
    fn_freq : name of the frequency file
    fn_kpath_def : optional (only for plotting later), special points definition
        file, see notes below
    
    returns:
    --------
    path_norm, freqs, special_points_path
    path_norm : array (nks,), sequence of cumulative norms of the difference
        vectors which connect each two adjacent k-points
    freqs : array (nks, nbnd), array with `nbnd` frequencies for each k-point,
        nbnd should be = 3*natom (natom = atoms in the unit cell)
    special_points_path : SpecialPointsPath instance or None

    notes:
    ------
    matdyn.x must have been instructed to calculate a phonon dispersion along a
    predefined path in the BZ. e.g. natom=2, nbnd=6, 101 k-points on path
        
        example:        
        -------------------------------------------------------------
        &input
            asr='simple',  
            amass(1)=26.981538,
            amass(2)=14.00674,
            flfrc='fc',
            flfrq='matdyn.freq.disp'
        /
        101                                | nks
        0.000000    0.000000    0.000000   |
        0.037500    0.037500    0.000000   | List of nks = 101 k-points
        ....                               |
        -------------------------------------------------------------


    `fn_freq` has the form
        <header>
        <k-point, (3,)>
        <frequencies,(nbnd,)
        <k-point, (3,)>
        <frequencies,(nbnd,)
        ...

        example:        
        -------------------------------------------------------------
        &plot nbnd=   6, nks= 101 /
                  0.000000  0.000000  0.000000
          0.0000    0.0000    0.0000  456.2385  456.2385  871.5931
                  0.037500  0.037500  0.000000
         23.8811   37.3033   54.3776  455.7569  457.2338  869.8832
        ... 
        -------------------------------------------------------------
    
    `fn_kpath_def` : 
        <coordinate of special point> #<name>
        ...

        example:
        -------------------------------------------------------------
        0    0    0     # $\Gamma$
        0.75 0.75 0     # K
        1 0.5 0         # W
        1 0 0           # X
        0 0 0           # $\Gamma$
        .5 .5 .5        # L
        -------------------------------------------------------------
    Note that you can put special matplotlib math text in this file. Everything
    after `#' is treated as a Python raw string.

    For correct plotting, the k-points defined in `fn_kpath_def` MUST of course
    be on the exact same k-path as the k-points listed in `fn_freq`.
    """
    from pwtools.kpath import SpecialPointsPath, SpecialPoint, get_path_norm
    ks, freqs = read_matdyn_freq(fn_freq)
    # parse k-path definition file
    if fn_kpath_def is not None:
        special_points = []
        fhk = open(fn_kpath_def)
        for line in fhk:    
            spl = line.strip().split()
            special_points.append(
                kpath.SpecialPoint(np.array(spl[:3], dtype=float), 
                    r'%s' %spl[-1].replace('#', '')))
        fhk.close()
        special_points_path = SpecialPointsPath(sp_lst=special_points)
    else:
        special_points_path = None
    # calculate path norms (= x-axis for plotting)
    path_norm = get_path_norm(ks)
    return path_norm, freqs, special_points_path


def ibrav2cell(ibrav, celldm):
    """Convert PWscf's ibrav + celldm to cell. All formulas are taken straight
    from the PWscf homepage. Don't blame me for errors. Use after testing.

    This function generates *primitive* cells. Note that in crys.py (and
    anywhere else in the package, for that matter) we do not have a distinction
    between conventional/primitive cell. We always think in primitive cells.
    Especially celldm in crys.py can be converted to/from `cell` and
    `cryst_const`. But here, `celldm` is the PWscf style celldm, which
    describes the *conventional* cell. For example, for an fcc cell (ibrav=2),
    celldm[0] == a == alat is the lattice constant "a" of the cubic
    conventional cell (cell=a*identity(3)), which is also found in a .cif file
    together with all symmetries. OTOH, for a hexagonal cell (ibrav=4)
    primitive == conventional cell.
    
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
    * ibrav = 14 is actually the only case where all 6 entries of `celldm` are
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
