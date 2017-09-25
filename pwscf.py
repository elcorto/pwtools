# pwscf.py
#
# Some handy tools to construct strings for building pwscf input files.
# Readers for QE postprocessing tool output (matdyn.x, dynmat.x).

import re, os, warnings
import numpy as np
from pwtools.common import fix_eps, str_arr, file_readlines, pj
from pwtools import parse, crys, common
from pwtools.num import EPS
from math import sin, acos, sqrt

def atpos_str(symbols, coords, fmt="%.16e", zero_eps=None, eps=EPS, delim=4*' '):
    """Convenience function to make a string for the ATOMIC_POSITIONS section
    of a pw.x input file.
    
    Parameters
    ----------
    symbols : sequence
        strings with atom symbols, (natoms,), must match with the
        rows of coords
    coords : array (natoms, 3) 
        with atomic coords, can also be (natoms, >3) to add constraints on
        atomic forces in PWscf
    eps : float
        Print values as 0.0 where abs(coords[i,j]) < eps. If eps < 0.0, then
        disable this.
    delim : str
        delimiter between columns

    Returns
    -------
    string

    Examples
    --------
    >>> print atpos_str(['Al', 'N'], array([[0,0,0], [0,0,1.]]))
    Al      0.0000000000    0.0000000000    0.0000000000
    N       0.0000000000    0.0000000000    1.0000000000
    """
    if zero_eps is not None:
        warnings.warn("`zero_eps` is deprecated, use `eps` > 0 instead",
                      DeprecationWarning)
    coords = np.asarray(coords)
    assert len(symbols) == coords.shape[0], "len(symbols) != coords.shape[0]"
    txt = '\n'.join("%s%s%s" %(symbols[i], delim, str_arr(row, fmt=fmt, eps=eps,
                                                          delim=delim)) \
        for i,row in enumerate(coords))
    return txt        


def atpos_str_fast(symbols, coords, work=None):
    """Fast version of atpos_str() for usage in loops. We use a fixed string
    dtype ``|S20`` to convert the array `coords` to string form. We also avoid
    all assert's etc for speed.
    
    Parameters
    ----------
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of coords
    coords : array (natoms, 3) with atomic coords, can also be (natoms, >3) to
        add constraints on atomic forces in PWscf
    work : optional, array of shape (natoms, 4)
        Pre-allocated work array. This can be the result of numpy.empty(). It
        is used to temporarily store the string dtype array.

    Returns
    -------
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
    
    Parameters
    ----------
    symbols : sequence of strings with atom symbols, (natoms,)
    masses : sequence if floats (natoms,) w/ atom masses
    pseudos : sequence of strings (natoms,) w/ pseudopotential file names

    Returns
    -------
    string

    Examples
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


def kpoints_str(lst, base='nk'):
    """[3,3,3] -> "nk1=3,nk2=3,nk3=3" 
    
    Useful for QE's phonon toolchain ph.x, q2r.x, matdyn.x
    """
    return ','.join(['%s%i=%i' %(base, i+1, x) for i, x in enumerate(lst)])
kpointstr = kpoints_str


def kpoints_str_pwin(lst, shift=[0,0,0]):
    """[3,3,3] -> " 3 3 3 0 0 0" 
    Useful for pwscf input files, card K_POINTS.

    Parameters
    ----------
    lst : sequence (3,)
    shift : sequence (3,), optional
    """
    return ' '.join(map(str, lst+shift))
kpointstr_pwin = kpoints_str_pwin


def kpoints_str_pwin_full(lst, shift=[0,0,0], gamma=True):
    """Full k-points string for pw.x input files, card K_POINTS.
    
    Parameters
    ----------
    lst : sequence (3,)
    shift : sequence (3,), optional
    gamma : bool, optional
        If lst == [1,1,1] then return "K_POINTS gamma", else
        "K_POINTS automatic <newline> 1 1 1 <`shift`>".
    """
    lst = lst if type(lst) == type('s') else list(lst) 
    if (lst == [1,1,1] and gamma) or (lst == 'gamma'):
        return "K_POINTS gamma"
    else:
        return "K_POINTS automatic\n%s"  %kpointstr_pwin(lst, shift=shift)
kpointstr_pwin2 = kpoints_str_pwin_full


def bool2str(x):
    """Return Fortran bool string for bool input."""
    return '.true.' if x else '.false.'


def read_matdyn_modes(filename, natoms=None):
    """Parse modes file produced by QE's matdyn.x.
    
    Parameters
    ----------
    filename : str
        File to parse (usually "matdyn.modes")
    natoms : int
        Number of atoms.
    
    Returns
    -------
    qpoints, freqs, vecs
    qpoints : 2d array (nqpoints, 3)
        All qpoints on the grid.
    freqs : 2d array, (nqpoints, nmodes) where nmodes = 3*natoms
        Each row: 3*natoms phonon frequencies in [cm^-1] at each q-point.
    vecs : 4d complex array (nqpoints, nmodes, natoms, 3)
        Complex eigenvectors of the dynamical matrix for each q-point.
    
    Examples
    --------
    >>> qpoints,freqs,vecs=read_matdyn_modes('matdyn.modes',natoms=27)
    # how many q-points? -> 8
    >>> qpoints.shape
    (8,3)
    # 1st q-point in file, mode #3 (out of 3*27) -> vectors on all 27 atoms
    >>> vecs[0,2,...].shape
    (27,3)
    # 1st q-point in file, mode #3, vector on atom #15
    >>> vecs[0,2,14,:].real
    array([-0.010832,  0.026063, -0.089511])
    >>> vecs[0,2,14,:].imag
    array([ 0.,  0.,  0.])

    Notes
    -----
    The file to be parsed looks like this::

           diagonalizing the dynamical matrix ...
      
       q =       0.0000      0.0000      0.0000
       **************************************************************************
           omega( 1) =     -26.663631 [THz] =    -889.402992 [cm-1]
       ( -0.218314   0.000000    -0.025643   0.000000    -0.116601   0.000000   )
       ( -0.086633   0.000000     0.108966   0.000000    -0.066513   0.000000   )
      [... natoms lines: x_real x_imag y_real y_imag z_real z_imag ... until
       next omega ...]
           omega( 2) =     -16.330246 [THz] =    -544.718372 [cm-1]
       (  0.172149   0.000000     0.008336   0.000000    -0.121991   0.000000   )
       ( -0.061497   0.000000     0.003782   0.000000    -0.018304   0.000000   )
      [... until omega(3*natoms) ...]
       **************************************************************************
           diagonalizing the dynamical matrix ...
      
      [... until next q-point ...]
       q =       0.0000      0.0000     -0.5000
       **************************************************************************
           omega( 1) =     -24.881828 [THz] =    -829.968443 [cm-1]
       ( -0.225020   0.000464    -0.031584   0.000061    -0.130217   0.000202   )
       ( -0.085499   0.000180     0.107383  -0.000238    -0.086854   0.000096   )
      [...]
       **************************************************************************
    """
    assert natoms is not None
    cmd = r"grep 'q.*=' %s | sed -re 's/.*q\s*=(.*)/\1/'" %filename
    qpoints = parse.arr2d_from_txt(common.backtick(cmd))
    nqpoints = qpoints.shape[0]
    nmodes = 3*natoms
    cmd = r"grep '^[ ]*(' %s | sed -re 's/^\s*\((.*)\)/\1/g'" %filename
    # vecs_file_flat: (nqpoints * nmodes * natoms, 6)
    # this line is the bottleneck
    vecs_file_flat = parse.arr2d_from_txt(common.backtick(cmd))
    vecs_flat = np.empty((vecs_file_flat.shape[0], 3), dtype=complex)
    vecs_flat[:,0] = vecs_file_flat[:,0] + 1j*vecs_file_flat[:,1]
    vecs_flat[:,1] = vecs_file_flat[:,2] + 1j*vecs_file_flat[:,3]
    vecs_flat[:,2] = vecs_file_flat[:,4] + 1j*vecs_file_flat[:,5]
    vecs = vecs_flat.flatten().reshape(nqpoints, nmodes, natoms, 3)
    cmd = r"grep omega %s | sed -re \
            's/.*omega.*=.*\[.*=(.*)\s*\[.*/\1/g'" %filename
    freqs = np.fromstring(common.backtick(cmd), sep=' ').reshape((nqpoints, nmodes))
    return qpoints, freqs, vecs


def read_dyn(filename, natoms=None):
    """Read one dynamical matrix file (for 1 qpoint) produced by ``ph.x`` and
    extract the same as :func:`read_matdyn_modes` for this qpoint only. 
    
    All arrays have one dim less compared to :func:`read_matdyn_modes`.
    
    Parameters
    ----------
    filename : str
        Name of dyn file. Example: "ph.dyn3" for qpoint 3.
    natoms : int
        number of atoms in the cell (used for nmodes=3*natoms only)
    
    Returns
    -------
    qpoints, freqs, vecs
    qpoints : 1d array (3,)
        The qpoint of the dyn file.
    freqs : 1d array, (nmodes,) where nmodes = 3*natoms
        3*natoms phonon frequencies in [cm^-1] at the q-point.
    vecs : 3d complex array (nmodes, natoms, 3)
        Complex eigenvectors of the dynamical matrix for the q-point.
    """
    assert natoms is not None
    cmd = r"egrep 'q.*=.*\(' %s | tail -n1 | sed -re 's/.*q\s*=.*\((.*)\)/\1/'" %filename
    qpoints = np.fromstring(common.backtick(cmd), sep=' ')
    assert qpoints.shape == (3,)
    nmodes = 3*natoms
    cmd = r"grep -v 'q.*=' %s | grep '^[ ]*(' | sed -re 's/^\s*\((.*)\)/\1/g'" %filename
    # vecs_file_flat: (nmodes * natoms, 6)
    # this line is the bottleneck
    vecs_file_flat = parse.arr2d_from_txt(common.backtick(cmd))
    vecs_flat = np.empty((vecs_file_flat.shape[0], 3), dtype=complex)
    vecs_flat[:,0] = vecs_file_flat[:,0] + 1j*vecs_file_flat[:,1]
    vecs_flat[:,1] = vecs_file_flat[:,2] + 1j*vecs_file_flat[:,3]
    vecs_flat[:,2] = vecs_file_flat[:,4] + 1j*vecs_file_flat[:,5]
    vecs = vecs_flat.flatten().reshape(nmodes, natoms, 3)
    cmd = r"grep omega %s | sed -re \
            's/.*omega.*=.*\[.*=(.*)\s*\[.*/\1/g'" %filename
    freqs = np.fromstring(common.backtick(cmd), sep=' ')
    return qpoints, freqs, vecs


def read_all_dyn(path, nqpoints=None, natoms=None, base='ph.dyn'):
    """Same as :func:`read_matdyn_modes()`, but instead of the file
    ``matdyn.modes`` which contains freqs,vecs for all qpoints, we read all
    dynamical matrix files in `path`, one per qpoint.

    Parameters
    ----------
    path : str
        Path where dyn files live.
    nqpoints : int
        number of dyn files (e.g. 5 for "ph.dyn1", ..., "ph.dyn5" if
        ``base='ph.dyn'``)
    natoms : int
    base : str
        Basename of the dyn files.
    
    Returns
    -------
    (qpoint,freqs,vecs)
        Same as :func:`read_matdyn_modes`
    """
    nmodes = 3*natoms
    qpoints = np.empty((nqpoints,3), dtype=float)
    freqs = np.empty((nqpoints, nmodes), dtype=float)
    vecs = np.empty((nqpoints, nmodes, natoms, 3), dtype=complex)
    for iq in range(nqpoints):
        filename = os.path.join(path, '%s%i' %(base, iq+1))
        qq, ff, vv = read_dyn(filename, natoms=natoms)
        qpoints[iq,...] = qq
        freqs[iq,...] = ff
        vecs[iq,...] = vv
    return qpoints, freqs, vecs


def read_dynmat(path='.', natoms=None, filename='dynmat.out', axsf='dynmat.axsf'):
    """Read ``dynmat.x`` output.

    `freqs` are parsed from `filename` and `vecs` from `axsf`. `qpoints` is
    alawys Gamma, i.e. [0,0,0].     

    Output format is the same as in :func:`read_dyn`.

    Parameters
    ----------
    path : str
        path where output files are
    natoms : int
    filename : str
        Text output from dynmat.x, where the frequencies are printed, relative
        to `path`.
    axsf : str
        AXSF file (``filxsf`` in input) with mode vectors as forces.
    
    Returns
    -------
    qpoints, freqs, vecs
    qpoints : 1d array (3,)
        The qpoint, which is Gamma, i.e. [0,0,0]
    freqs : 1d array, (nmodes,) where nmodes = 3*natoms
        3*natoms phonon frequencies in [cm^-1] at the q-point.
    vecs : 3d real array (nmodes, natoms, 3)
        Real parts (???) if the eigenvectors of the dynamical matrix for the
        q-point.
    
    Notes
    -----
    We assume the output to be generated with ``dynmat.x < dynmat.in >
    dynmat.out``. 
    """
    assert natoms is not None, ("natoms is None")
    nmodes = 3*natoms
    out_fn = pj(path, filename)
    axsf_fn = pj(path, axsf)
    cmd = "grep -A{0} PRIMCO {1} | sed -re '/PRIMCO.*/{{N;d;}}' | \
            awk '{{print $5\" \"$6\" \"$7}}'".format(natoms+1, axsf_fn)
    qpoints = np.zeros((3,))
    vecs = np.fromstring(common.backtick(cmd), sep=' ').reshape(nmodes,natoms,3)
    cmd = "grep -A{0} 'mode.*cm-1' {1} | grep -v mode | \
           awk '{{print $2}}'".format(nmodes, out_fn)
    freqs = np.fromstring(common.backtick(cmd), sep=' ')
    return qpoints,freqs,vecs


def read_dynmat_ir_raman(filename='dynmat.out', natoms=None, 
                         cols={1: 'freqs', 3:'ir', 4: 'raman', 5: 'depol'}):
    """Read ``dynmat.x`` text output file and extract IR and Raman
    intensities.
    
    Parameters
    ----------
    filename : str
        dynmat.x text output file (e.g. from ``dynmat.x < dynmat.in >
        dynmat.out``)
    natoms : int
        number of atoms in the cell
    cols : dict
        column numbers of the text block

    Returns
    -------
    cols = None 
        Return the parsed array as found in the file
    cols = dict 
        Return dict with keys from `cols` and 1d arrays ``{'freqs': <array>,
        'ir': <array>, 'raman': <array>, 'depol': <array>}``. If a column is
        not present, the array is None.
 
    Notes
    -----
    The parsed textblock looks like this::

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
    
    Some columns (e.g. IR, Raman) may be missing.
    """                    
    assert natoms is not None, ("natoms is None")
    cmd = "grep -A{0} 'mode.*cm-1' {1} | grep -v mode".format(3*natoms, filename)
    arr = parse.arr2d_from_txt(common.backtick(cmd))
    if cols is None:
        return arr
    else:
        dct = {}
        for ii,name in cols.items():
            if arr.shape[1] >= (ii+1):
                dct[name] = arr[:,ii]
            else:
                dct[name] = None
        return dct


def read_dynmat_out(*args, **kwds):
    """Backward compat wrapper for :func:`read_dynmat_ir_raman`."""
    warnings.warn("read_dynmat_out() is deprecated, use read_dynmat_ir_raman()",
                  DeprecationWarning)
    return read_dynmat_ir_raman(*args, **kwds)                  


def read_matdyn_freq(filename):
    """Parse frequency file produced by QE's matdyn.x ("flfrq" in matdyn.x
    input, usually "matdyn.freq" or so) when calculating a phonon dispersion on
    a grid (ldisp=.true., used for phonon dos) or a pre-defined k-path in the
    BZ.

    In QE 5.x, a file with suffix ".gp" (e.g. "matdyn.freq.gp") is now written,
    where::
    >>> import numpy as np
    >>> from pwtools import kpath, pwscf
    >>> d = np.loadtxt("matdyn.freq.gp")
    >>> kpoints,freqs = pwscf.read_matdyn_freq("matdyn.freq")
    >>> allclose(d[:,0], kpath.get_path_norm(kpoints))
    >>> allclose(d[:,1:], freqs)
    
    Parameters
    ----------
    filename : file with k-points and phonon frequencies
    
    Returns
    -------
    kpoints : array (nks, 3)
        Array with `nks` k-points. AFAIK the unit is always ``2*pi/alat`` with
        ``alat = celldm(1)``.
    freqs : array (nks, nbnd)
        Array with `nbnd` energies/frequencies at each of the `nks` k-points.
        For phonon DOS, nbnd == 3*natoms.
    
    Notes
    -----
    `matdyn.in`::

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

    `filename` has the form::

        <header>
        <k-point, (3,)>
        <frequencies,(nbnd,)
        <k-point, (3,)>
        <frequencies,(nbnd,)
        ...
    
    for example::

        &plot nbnd=   6, nks= 101 /
                  0.000000  0.000000  0.000000
          0.0000    0.0000    0.0000  456.2385  456.2385  871.5931
                  0.037500  0.037500  0.000000
         23.8811   37.3033   54.3776  455.7569  457.2338  869.8832
         .....

    See Also
    --------
    bin/plot_dispersion.py, :func:`pwtools.kpath.plot_dis`,
    :func:`pwtools.kpath.get_path_norm`
    """
    lines = file_readlines(filename)
    # Read number of bands (nbnd) and k-points (nks). OK, Fortran's namelists
    # win here :)
    # nbnd: number of bands = number of frequencies per k-point = 3*natoms for
    #   phonons
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

    Parameters
    ----------
    ibrav : int
        1 ... 14
    celldm : sequence of length 6
        This not the isame length 6 array `celldm` in crys.py. Here, the
        entries which are not needed can be None.
    
    Returns
    -------
    array (3,3) : cell vectors as rows, unit is that of celldm[0], i.e. a

    Notes
    -----
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
        # v1 = (a/2)(-1,0,1),  v2 = (a/2)(0,1,1), v3 = (a/2)(-1,1,0)
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
        raise Exception("illegal ibrav: %s" %ibrav)
    return cell        
