#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# vim:ts=4:sw=4:et

# 
# Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
# The pydos package. 
# 

"""
Parse and post-process molecular dynamics data produced by the Quantum
Espresso package (quantum-espresso.org). 

Currently, pw.x and "calculation='md'" type data is supported (e.g. NOT
'vc-md').  Other calulation types write results in different order to the
outfile. Since we exploit the knowledge of the order for performance reasons,
supporting different orders requires a little rewrite/generalisation of the
code in parse_pwout().

Tested with QE 3.2.3 and 4.0.1. 

TODO
----
- Drop some line.strip(), extend REs where possible to use faster re.match()
  instead of re.search(). But possibly low performance gain since re.* is alraedy
  very fast.

- Turn on Fortran order by default in array constructors (or wherever) in
  numpy to avoid copies by f2py extensions which operate on rank>2 arrays. 

  Changing ndarray.__new__() doesn't help, b/c it's only called in cases `a
  = ndarray()`, i.e. like __init__() -- call class constructor directly.
  empty(), array(), etc. are no members of ndarray. They are normal
  numpy.core.multiarray functions with kwarg "order={'C','F'}".

  => Best would be to subclass numpy.ndarray and redefind array() etc.

- Make a Parser class which holds all parsing functions for in- and outfile.
  The class also holds all arrays constructed from parsing (R, T, P) and also
  all parsing results which are dicts --> no need to pass them around as
  function args. 

- class Array with read, write methods ...

- Optional spline interpolation of R trajectories for each atom i and
  coordinate k: R[i,:,k]

- Test suite. We have some primitive doctests now but hey ...

- setup.py, real pythonic install, no fiddling with PYTHONPATH etc,
  you can also install in $HOME dir using `setup.py install
  --prefix=~/some/path/`, but it's probably overkill for this little project

- {write|read}bin_array(): use new numpy.save(), numpy.load() -> .npy/.npz
  format


Units
------
From http://www.quantum-espresso.org/input-syntax/INPUT_PW.html

    All quantities whose dimensions are not explicitly specified are in
    RYDBERG ATOMIC UNITS

    See constants.py

File handling:
--------------
Some functions accept only filenames or only fileobjects. Some accept both.
Default in all functions is that 
- in case of filenames, a file will be opened
  and closed inside the function. 
- In case of fileobjects, they will NOT be closed
  but returned, i.e. they must be opened and closed outside of a function.
- In case they accept both, the file will be closed inside in any case.  

Calculation of the PDOS
-----------------------

PDOS obtained the VACF way (V -> VACF -> FFT(VACF)) and the direct way
(direct_pdos()) differ a bit. Dunno why yet. But using different FFT algos
for the two methods changes nothing (very good), i.e. the numerical
difference between scipy.fftpack.fft() and dft() (and dft_axis()) are much much
smaller.

assert -> _assert()
-------------------
:%s/assert[ ]*\(.*\)[ ]*,[ ]*[(]\?\([ `'"a-zA-Z0-9]\+\)[)]\?/_assert(\1, \2)/gc
"""

from debug import Debug
DBG = Debug()

# timing of the imports
##DBG.t('import')

import re
import math
import sys
import os
import types
import subprocess as S
import ConfigParser
from os.path import join as pjoin

import numpy as np
# faster import, copied file from scipy sources
##from scipy.io.npfile import npfile
from scipy_npfile import npfile

# slow import time !!!
from scipy.fftpack import fft, ifft
from scipy.linalg import inv
from scipy.integrate import simps

# own modules
import constants
import _flib
from common import assert_cond as _assert

##DBG.pt('import')

#-----------------------------------------------------------------------------
# globals 
#-----------------------------------------------------------------------------

VERBOSE=True
VERSION = '0.2'

# All card names that may follow the namelist section in a pw.x input file.
INPUT_PW_CARDS = [\
    'atomic_species',
    'atomic_positions',
    'k_points',
    'cell_parameters',
    'occupations',
    'climbing_images',
    'constraints',
    'collective_vars']

#-----------------------------------------------------------------------------
# file handling
#-----------------------------------------------------------------------------

def fileo(val, mode='r'):
    """Return open file object with mode `mode`.

    args:
    -----
    val : str or file object
    mode : file mode (everything that Python's open() can handle)
    """
    if isinstance(val, types.StringType):
        return open(val, mode)
    elif isinstance(val, types.FileType):
        if val.closed:
            raise StandardError("need an open file") 
        elif not val.mode == mode:
            raise ValueError("mode of file '%s' is '%s', must be '%s'" 
                %(val.name, val.mode, mode))
        return val
    else:
        raise ValueError("`val` must be string or file object")

#-----------------------------------------------------------------------------

# This should become a general save-nd-array-to-txt-file function. We can only write 
# 1d or 2d arrays to file with np.savetxt. `axes` specifies the axes which form
# the 2d arrays. Loop over all permutations of the remaining axes and write all
# 2d arrays to open file. For the permutations to be general, we want nested
# loops of variable depth, i.e. we must use recursion. This would be also nice
# for scell_mask(), which also uses hard-coded nested loops of fixed depth (3
# there) now. Although it's not necessary there (no 4-dim crystals :)), it
# would be nice to know how it's done.

##def writetxtnd(fn, a, axes=(0,1)):
##    _assert(lan(axes)==2, "`axes` must be length 2")
##    remain_axes=[]
##    for ax in range(a.ndim):
##        if ax not in axes:
##            remain_axes.append(ax)
##    fileo=open(fn, 'w')
##    sl = [0]*a.ndim
##    for ax in axes:
##        sl[ax] = slice(None)
##    for ax in remain_axes:
##        for index in a.shape[ax]:
##            np.savetxt(fileo, slicetake())

# quick hack for 3d arrays
def writetxt(fl, arr, axis=-1):
    maxdim=3
    _assert(arr.ndim <= maxdim, 'no rank > 3 arrays supported')
    fl = fileo(fl, 'w')
    # 1d and 2d case
    if arr.ndim < maxdim:
        np.savetxt(fl, arr)
    # 3d        
    else:        
        sl = [slice(None)]*arr.ndim
        for ind in range(arr.shape[axis]):
            sl[axis] = ind
            np.savetxt(fl, arr[sl])
    fl.close()        
    
#-----------------------------------------------------------------------------

def writearr(fl, arr, order='C', endian='<', comment=None, info=None,
             type='bin', axis=-1):
    """Write `arr` to binary (*.dat) or text file (*.txt) `fl` and also save
    the shape, endian etc.  in a cfg-style file "`fl`.info".

    args:
    -----
    arr : numpy ndarrray
    fl : str or open fileobject
    comment : string
        a comment which will be written to the .info file (must start with '#')
    info : dict of dicts 
        addtional sections for the .info file        
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
    only type == 'bin'
        order : str, {'C','F'}
            'C' - row major, 'F' - column major
        endian : str, {'<', '>'}
            '<' - little, '>' - big
    only type == 'txt'
        axis : axis kwarg for writetxt()
    """
    _assert(type in ['bin', 'txt'], "`type` must be 'bin' or 'txt'")
    if isinstance(fl, types.StringType):
        fname = fl
    else:
        fname = fl.name
    verbose("[writearr] writing: %s" %fname)
    verbose("[writearr]     shape: %s" %repr(arr.shape))
    if type == 'bin':
        # here, perm could be anything, will be changed in npfile() anyway
        perm = 'wb'
        fl = fileo(fl, mode=perm)
        npf = npfile(fl, order=order, endian=endian, permission=perm)
        npf.write_array(arr)
        # closes also `fl`
        npf.close()
    else:
        writetxt(fl, arr, axis=axis)
    
    # --- .info file ------------------
    c = PydosConfigParser()
    sec = 'array'
    c.add_section(sec)
    c.set(sec, 'shape', tup2str(arr.shape))
    if type == 'bin':
        c.set(sec, 'order', order)
        c.set(sec, 'endian', endian)
        c.set(sec, 'dtype', str(arr.dtype))
    if info is not None:
        c = _add_info(c, info) 
    f = open(fname + '.info', 'w')
    if comment is not None:
        print >>f, comment
    c.write(f)
    f.close()

#-----------------------------------------------------------------------------

def _add_info(config, info):
    for sec, dct in info.iteritems():
        config.add_section(sec)
        for key, val in dct.iteritems():
            config.set(sec, key, val)
    return config


#-----------------------------------------------------------------------------

def readbin(fn):
    """Read binary file `fn` array according the information in
    in a txt file "`fn`.shape".

    args
    -----
    fn : str
    
    returns:
    --------
    numpy ndarray 
    """
    verbose("[readbin] reading: %s" %fullpath(fn))
    c = PydosConfigParser()
    f = open(fn + '.info')
    c.readfp(f)
    f.close()
    sec = 'array'
    shape = str2tup(c.get(sec, 'shape'))
    order = c.get(sec, 'order')
    endian = c.get(sec, 'endian')
    dtype = np.dtype(c.get(sec, 'dtype'))
    
    npf = npfile(fn, order=order, endian=endian, permission='rb')
    arr = npf.read_array(dtype, shape=shape)
    npf.close()
    verbose("[readbin]     shape: %s" %repr(arr.shape))
    return arr



#-----------------------------------------------------------------------------
# parsing 
#-----------------------------------------------------------------------------

def toslice(val):
    """A simple wrapper around numpy.s_() taking strings as argument. 
    Convert strings representing Python/numpy slice to slice
    objects.
    
    args:
    -----
    val : string

    examples:
    ---------
    '3'     -> 3
    '3:'    -> slice(3, None, None)
    '3:7'   -> slice(3, 7, None)
    '3:7:2' -> slice(3, 7, 2)
    '3::2'  -> slice(3, None, 2)
    '::2'   -> slice(None, None, 2)
    '::-1'  -> slice(None, None, -1)

    >>> import numpy as np
    >>> np.s_[1:5]
    slice(1, 5, None)
    >>> toslice('1:5')
    slice(1, 5, None)
    """
    _assert(isinstance(val, types.StringType), "input must be string")
    # This eval() trick works but seems hackish. Better ideas, anyone?
    return eval('np.s_[%s]' %val)

#-----------------------------------------------------------------------------

def tobool(val):
    """Convert `val` to boolean value True or False.
        
    args:
    -----
    val : bool, string, integer
        '.true.', '1', 'true',  'on',  'yes', integers != 0 -> True
        '.false.','0', 'false', 'off', 'no',  integers == 0 -> False
    
    returns:
    --------
    True or False

    notes:
    ------
    All string vals are case-insensitive.
    """
    if isinstance(val, types.BooleanType):
        if val == True:
            return True
        else:
            return False
    got_str = False
    got_int = False
    if isinstance(val, types.StringType):
        got_str = True
        val = val.lower()
    elif isinstance(val, types.IntType):
        got_int = True
    else:
        raise StandardError, "input value must be string or integer"
    if (got_str and (val in ['.true.', 'true', 'on', 'yes', '1'])) \
        or (got_int and (val != 0)):
        ret = True
    elif (got_str and (val in ['.false.', 'false', 'off', 'no', '0'])) \
        or (got_int and (val == 0)):
        ret = False
    else:
        raise StandardError("illegal input value '%s'" %repr(val))
    return ret

#-----------------------------------------------------------------------------

def _float(st):
    """Convert strings representing numbers to Python floats using
    float(). The returned value is a double (or whatever the float() of your
    Python installation  returns). 
    
    Especially, strings representing Fortran floats are handled. Fortran Reals
    (= single) are converted to doubles. Kind parameters (like '_10' in
    '3.0d5_10') are NOT supported, they are ignored.

    args:
    -----
    st : string

    returns:
    --------
    float
    """
    _assert(isinstance(st, types.StringType), "`st` must be string")
    st = st.lower()
    if not 'd' in st:
        return float(st)
    else:
        # >>> s='  40.0d+02_10  '
        # >>> m.groups()
        # ('40.0', '+', '02', '_10  ')
        # >>> s='  40.0d02  '
        # >>> m.groups()
        # ('40.0', '', '02', '  ')
        #
        rex = re.compile(r'[ ]*([0-9\.]+)d([+-]*)([0-9]+)([_]*.*)')
        m = rex.match(st)
        if m is None:
            raise ValueError("no match on string '%s'" %st)
        if m.group(4).strip() != '':
            verbose("[_float] WARNING: skipping kind '%s' in string '%s'" 
                %(m.group(4), st))
        ss = "%se%s%s" %m.groups()[:-1]
        return float(ss)

#-----------------------------------------------------------------------------

def _repr(var, float_fmt="%.15e"):
    """Similar to Python's repr(), but return floats formated
    with `float_fmt`. Python's repr() handles also var = None.
    
    args:
    -----
    var : almost anything (str, None, int, float)
    
    examples:
    ---------
    1     -> '1'
    1.0   -> '1.000000000000000e+00' 
    None  -> 'None'
    'abc' -> 'abc' (repr() does: 'abc' -> "'abc'")
    """
    if isinstance(var, types.FloatType):
        return float_fmt %var
    elif isinstance(var, types.StringType):
        return var
    else:
        return repr(var)

#-----------------------------------------------------------------------------

def tup2str(t):
    """
    (1,2,3) -> "1 2 3"
    """
    return " ".join(map(str, t))

#-----------------------------------------------------------------------------

def str2tup(s, func=int):
    """
    "1 2 3" -> (func('1'), func('2'), func('3')) 
    """
    return tuple(map(func, s.split()))

#-----------------------------------------------------------------------------

class PydosConfigParser(ConfigParser.SafeConfigParser):
    """All values passed as `arg` to self.set(self, section, option, arg) are
    converted to a string with _repr(). get*() methods are the usual ones
    provided by the base class ConfigParser.SafeConfigParser: get(), getint(),
    getfloat(), getboolean(). Option keys are case-sensitive.
    """
    # make keys case-sensitive
    ConfigParser.SafeConfigParser.optionxform = str
    def set(self, section, option, arg):
        ConfigParser.SafeConfigParser.set(self, section, option, _repr(arg))

#-----------------------------------------------------------------------------

def next_line(fh):
    """
    Will raise StopIteration at end of file.
    """
##    try:
##        return fh.next().strip()
##    except StopIteration:
##        verbose("[next_line] End of file %s" %fh)
##        return None
    return fh.next().strip()

#-----------------------------------------------------------------------------

# cannot use:
#
#   >>> fh, flag = scan_until_pat(fh, ...)
#   # do something with the current line
#   >>> line = fh.readline()
#   <type 'exceptions.ValueError'>: Mixing iteration and read methods would
#   lose data
# 
# Must return line at file position instead.

def scan_until_pat(fh, pat="atomic_positions", err=True, retline=False):
    """
    Go to pattern `pat` in file `fh` and return `fh` at this position. Usually
    this is a header followed by a table in a pw.x input or output file.

    args:
    ----
    fh : file like object
    pat : string, pattern in lower case
    err : bool, raise error at end of file b/c pattern was not found
    retline : bool, return current line
    
    returns:
    --------
    fh : file like object
    flag : int
    {line : current line}
    
    notes:
    ------
    Currently all lines in the file are converted to lowercase before scanning.

    examples:
    ---------
    example file (pw.x input):
        
        [...]
        CELL_PARAMETERS
           4.477898982  -0.031121661   0.004594438
          -2.231322621   3.882493243  -0.004687159
           0.012261087  -0.006915066  12.050227607
        ATOMIC_SPECIES
        Si 28.0855 Si.LDA.fhi.UPF
        O 15.9994 O.LDA.fhi.UPF   
        Al 26.981538 Al.LDA.fhi.UPF
        N 14.0067 N.LDA.fhi.UPF
        ATOMIC_POSITIONS alat                              <---- fh at this pos
        Al       4.482670384  -0.021685570   4.283770714         returned
        Al       2.219608875   1.302084775   8.297440557
        Si      -0.015470487  -0.023393016   1.789590196
        Si       2.194751751   1.364416814   5.817547157
        [...]
    """
    for line in fh:
        if line.strip().lower().startswith(pat):
            if retline:
                return fh, 1, line
            return fh, 1
    if err:
        raise StandardError("end of file '%s', pattern "
            "'%s' not found" %(fh, pat))
    # nothing found = end of file
    if retline:
        return fh, 0, line
    return fh, 0

#-----------------------------------------------------------------------------

def scan_until_pat2(fh, rex, err=True, retmatch=False):
    """
    *Slightly* faster than scan_until_pat().

    args:
    ----
    fh : file like object
    rex : Regular Expression Object with compiled pattern for re.match().
    err : bool, raise error at end of file b/c pattern was not found
    retmatch : bool, return current Match Object
    
    returns:
    --------
    fh : file like object
    flag : int
    {match : re Match Object}

    notes:
    ------
    For this function to be fast, you must pass in a compiled re object. 
        rex = re.compile(...)
        fh, flag = scan_until_pat2(fh, rex)
    or 
        fh, flag = scan_until_pat2(fh, re.compile(...))
    BUT, if you call this function often (e.g. in a loop) with the same
    pattern, use the first form, b/c otherwise re.compile(...) is evaluated in
    each loop pass.
    """
    # Evaluating rex.match(line) twice if a match occured should be no
    # performance penalty, probably better than assigning m=rex.match(line)
    # in each loop. Could be a possible problem if called many times on short
    # files or a file with many matches.
    for line in fh:
        if rex.match(line) is not None:
            if retmatch:
                return fh, 1, rex.match(line)
            return fh, 1
    if err:
        raise StandardError("end of file '%s', pattern "
            "not found" %fh)
    # nothing found = end of file, rex.match(line) should be == None
    if retmatch:
        return fh, 0, rex.match(line)
    return fh, 0

#-----------------------------------------------------------------------------

# Must parse for ATOMIC_SPECIES and ATOMIC_POSITIONS separately (open, close
# infile) each time b/c the cards can be in arbitrary order in the input file.
# Therefore, we cant take an open fileobject as argumrent, but use the
# filename.
def atomic_species(fn):
    """Parses ATOMIC_SPECIES card in a pw.x input file.

    args:
    -----
    fn : filename of pw.x input file

    returns:
    --------
        {'atoms': atoms, 'masses': masses, 'pseudos': pseudos}
        
        atoms : list of strings, (number_of_atomic_species,), 
            ['Si', 'O', 'Al', 'N']
        masses : 1d arary, (number_of_atomic_species,)
            array([28.0855, 15.9994, 26.981538, 14.0067])
        pseudos : list of strings, (number_of_atomic_species,)
            ['Si.LDA.fhi.UPF', 'O.LDA.fhi.UPF', 'Al.LDA.fhi.UPF',
            'N.LDA.fhi.UPF']
    notes:
    ------
    scan for:
        [...]
        ATOMIC_SPECIES|atomic_species
        [possible empty lines]
        Si 28.0855 Si.LDA.fhi.UPF
        O 15.9994 O.LDA.fhi.UPF   
        Al 26.981538 Al.LDA.fhi.UPF
        N 14.0067 N.LDA.fhi.UPF
        [...]
    """
    verbose('[atomic_species] reading ATOMIC_SPECIES from %s' %fn)
    fh = open(fn)
    # rex: for the pseudo name, we include possible digits 0-9 
    rex = re.compile(r'[ ]*([a-zA-Z]+)[ ]+([0-9\.]+)[ ]+([0-9a-zA-Z\.]*)')
    fh, flag = scan_until_pat(fh, pat='atomic_species')
    line = next_line(fh)
    while line == '':
        line = next_line(fh)
    match = rex.match(line)
    lst = []
    while match is not None:
        # match.groups: tuple ('Si', '28.0855', 'Si.LDA.fhi.UPF')
        lst.append(list(match.groups()))
        line = next_line(fh)
        match = rex.match(line)
    # numpy string array :)
    ar = np.array(lst)
    atoms = ar[:,0].tolist()
    masses = np.asarray(ar[:,1], dtype=float)
    pseudos = ar[:,2].tolist()
    fh.close()
    return {'atoms': atoms, 'masses': masses, 'pseudos': pseudos}

#-----------------------------------------------------------------------------

def cell_parameters(fn):
    """Parses CELL_PARAMETERS card in a pw.x input file. Extract primitive
    lattice vectors.

    notes:
    ------
    From the PWscf help:
        
        --------------------------------------------------------------------
        Card: CELL_PARAMETERS { cubic | hexagonal }
        
        Optional card, needed only if ibrav = 0 is specified, ignored
        otherwise!
        
        Flag "cubic" or "hexagonal" specify if you want to look for symmetries
        derived from the cubic symmetry group (default) or from the hexagonal
        symmetry group (assuming c axis as the z axis, a axis along the x
        axis).
        
        v1, v2, v3  REAL

            Crystal lattice vectors:
            v1(1)  v1(2)  v1(3)    ... 1st lattice vector
            v2(1)  v2(2)  v2(3)    ... 2nd lattice vector
            v3(1)  v3(2)  v3(3)    ... 3rd lattice vector

            In alat units if celldm(1) was specified or in a.u. otherwise.
        --------------------------------------------------------------------
        
        In a.u. = Rydberg atomic units (see constants.py).
    """
    verbose('[cell_parameters] reading CELL_PARAMETERS from %s' %fn)
    fh = open(fn)
    rex = re.compile(r'[ ]*(([ ]*-*[0-9\.]+){3})[ ]*')
    fh, flag = scan_until_pat(fh, pat="cell_parameters")
    line = next_line(fh)
    while line == '':
        line = next_line(fh)
    match = rex.match(line)
    lst = []
    while match is not None:
        # match.groups(): ('1.3 0 3.0', ' 3.0')
        lst.append(match.group(1).strip().split())
        line = next_line(fh)
        match = rex.match(line)
    fh.close()
    cp = np.array(lst, dtype=float)
    _assert(cp.shape[0] == cp.shape[1], "dimentions of `cp` don't match")
    return cp


#-----------------------------------------------------------------------------

def atomic_positions(fn, atspec=None):
    """Parse ATOMIC_POSITIONS card in pw.x input file.
    
    args:
    -----
    fn : filename of pw.x input file
    atspec : optional, dict returned by atomic_species()
    
    returns:
    --------
        {'R0': R0, 'natoms': natoms, 'massvec': massvec, 'symbols':
        symbols, 'unit': unit}
        
        R0 : ndarray,  (natoms, 3)
        natoms : int
        massvec : 1d array, (natoms,)
        symbols : list of strings, (natoms,), ['Al', 'A', 'Si', ...]
        unit : string, 'alat', 'crystal', etc.

    notes:
    ------
    scan for:
        [...]
        ATOMIC_POSITIONS|atomic_positions [<unit>]
        [0 or more empty lines]
        Al       4.482670384  -0.021685570   4.283770714
        Al       2.219608875   1.302084775   8.297440557
        ...
        Si       2.134975048   1.275864192  -0.207552657
        [empty or nonempty line that does not match the RE]
        [...]

        <unit> is a string: 'alat', 'crystal' etc.
    """
    verbose("[atomic_positions] reading ATOMIC_POSITIONS from %s" %fn)
    if atspec is None:
        atspec = atomic_species(fn)
    rex = re.compile(r'[ ]*([a-zA-Z]+)(([ ]+-*[0-9\.]+){3})[ ]*')
    fh = open(fn)
    fh, flag, line = scan_until_pat(fh, pat="atomic_positions", retline=True)
    line = line.strip().lower().split()
    if len(line) > 1:
        unit = line[1]
    else:
        unit = ''
    line = next_line(fh)
    while line == '':
        line = next_line(fh)
    lst = []
    # Must use REs here b/c we don't know natoms. 
    match = rex.match(line)
    while match is not None:
        # match.groups():
        # ('Al', '       4.482670384  -0.021685570   4.283770714', '    4.283770714')
        lst.append([match.group(1)] + match.group(2).strip().split())
        line = next_line(fh)
        match = rex.match(line)
    ar = np.array(lst)
    symbols = ar[:,0].tolist()
    # same as R0 = np.asarray(ar[:,1:], dtype=float)
    R0 = ar[:,1:].astype(float)
    natoms = R0.shape[0]
    masses = atspec['masses']
    atoms = atspec['atoms']
    massvec = np.array([masses[atoms.index(s)] for s in symbols], dtype=float)
    fh.close()
    return {'R0': R0, 'natoms': natoms, 'massvec': massvec, 'symbols':
        symbols, 'unit': unit}

#-----------------------------------------------------------------------------

def atomic_positions_out(fh, rex, work):
    """Parse ATOMIC_POSITIONS card in pw.x output file.

    args:
    -----
    fh : open file (pw.x output file)
    rex : compiled regular expression object with pattern for rex.search()
    work : 2D array (natoms x 3)
    
    returns:
    --------
    fh
        
    usage:
    ------
    while ..
        fh = scan_until_pat(fh, ...)
        fh = atomic_positions_out(fh, r, w)
    
    notes:
    ------
    - scan for:
        [...]
        ATOMIC_POSITIONS|atomic_positions
        [0 or more empty lines]
        Al       4.482670384  -0.021685570   4.283770714
        Al       2.219608875   1.302084775   8.297440557
        ...
        Si       2.134975048   1.275864192  -0.207552657
        [empty or nonempty line that does not match the RE]
        [...]
    
    - With this implementstion, `rex` must be:
        
        With scan_until_pat*(), we need to know that we extract 3 numbers:
        >>> pat =  r'[ ]*[A-Za-z]+(([ ]+-*[0-9\.]+){3})'
        >>> rex = re.compile(pat)
    
        For scanning the whole file w/o the usage of scan_until_pat*() first,
        we have to know the atom symbols. We would use this kind of pattern if
        we'd parse the file with perl & friends:
        >>> atoms = ['Si', 'O', 'Al', 'N']
        >>> pat =  r'(%s)' %r'|'.join(atoms) + r'(([ ]+-*[0-9\.]+){3})'
        >>> rex = re.compile(pat)
        
    - Is a *little* bit slower than atomic_positions_out2.
    
    - In-place modification of `work`!!!!!
    """
    line = next_line(fh)
    while line == '':
        line = next_line(fh)
    c = -1
    match = rex.search(line)
    while match is not None:
        c += 1
        work[c,:] = np.array(match.group(2).strip().split()).astype(float)
        line = next_line(fh)
        match = rex.search(line)
    return fh

#-----------------------------------------------------------------------------

def atomic_positions_out2(fh, natoms, work):
    """Parse ATOMIC_POSITIONS card in pw.x output file.

    args:
    -----
    fh : open file
    natoms : number of atoms (i.e. number of rows of the table to read)
    work : 2D array (natoms x 3)
    
    returns:
    --------
    fh
        
    usage:
    ------
    while ..
        fh = scan_until_pat(fh, ...)
        fh = atomic_positions_out2(fh, n, w)
    
    notes:
    ------
    scan for:
        [...] 
        ATOMIC_POSITIONS|atomic_positions
        [0 or more empty lines]
        Al       4.482670384  -0.021685570   4.283770714    |
        Al       2.219608875   1.302084775   8.297440557    | natoms
        ...                                                 | rows
        Si       2.134975048   1.275864192  -0.207552657    |
        [anything else here]
        [...]
        
    In-place modification of `work`!!!!!
    """
    line = next_line(fh)
    while line == '':
        line = next_line(fh)
    # natoms count instead of RE matching        
    for i in xrange(natoms):
        work[i,:] = np.array(line.split()[1:]).astype(float)
        line = next_line(fh)
    return fh

#-----------------------------------------------------------------------------

def conf_namelists(f, cardnames=INPUT_PW_CARDS):
    """
    Parse "namelist" part of a pw.x input file.

    args:
    -----
    f : open fileobject or filename, if fileobject, it will not be closed

    notes:
    ------
    file to parse:
        [...]
        &namelist1
            FLOATKEY = 1.0d0, stringkey = 'foo' [,]
            boolkey = .true. [,]
            intkey = 10 [,]
            ...
        /
        &namelist2
            floatkey = 2.0d0, stringkey = 'bar' [,]
            boolkey = .false. [,]
            intkey = -10 [,]
            ...
        /
        ...

        [CARD SECTION]
        ATOMIC_SPECIES
        [...]
        
    
    returns: 
    --------
    dict of dicts:
        
        {namelist1:
            {'floatkey': '1.0d0', 'stringkey': "'foo'",
              boolkey = '.true.', intkey = '10'},
         namelist2:
            {'floatkey': '2.0d0', 'stringkey': "'bar'",
              boolkey = '.false.', intkey = '-10'},
        }
        
    All keys are converted to lowercase! All values are strings and must be
    converted.

        floatkey  = _float(d['namelist1']['floatkey'])
        stringkey = d['namelist1']['stringkey']
        intkey    = int(d['namelist1']['intkey'])
        boolkey   = tobool(d['namelist1']['boolkey'])
    """
    fh = fileo(f)
    verbose("[conf_namelists] parsing %s" %fh.name)
    dct = {}   
    for line in fh:
        # '   A = b, c=d,' -> 'A=b,c=d'
        line = line.strip().strip(',').replace(' ', '')
        if line.startswith('&'):
            # namelist key value pairs
            nl_kvps = []
            nl = line[1:].lower()
        elif line == '':
            line = f.next().strip().strip(',').replace(' ', '')
        elif line == '/':
            # nl = 'x', enter dict for namelist `nl` in `dct` under name 'x'.
            # [['a', 'b'], ['c', 'd']] -> dct = {'x': {'a': 'b', 'c': 'd'}, ...}
            dct[nl] = dict(nl_kvps)
        # end of namelist part
        elif line.lower() in cardnames:
            break
        else:
            # 'A=b,c=d' -> ['A=b', 'c=d'] -> 
            # nl_kvps = [..., ['a', 'b'], ['c', 'd']]
            for p in line.split(','):
                tmp = p.split('=')
                tmp[0] = tmp[0].lower()
                # HACK: remove nested quotes: "'foo'" -> 'foo'
                if tmp[1][0] == "'" and tmp[1][-1] == "'":
                    tmp[1] = eval(tmp[1])
                nl_kvps.append(tmp)
    return dct

#-----------------------------------------------------------------------------

# XXX finish me ....
class Array(object):
    """Class to represent arrays T, P, R, ...
    """
    def __init__(self, shape=None, timeaxis=0, sliceaxis=None):
        self.array = np.empty(shape, dtype=float)
        self.timeaxis = timeaxis
        if sliceaxis is None:
            self.sliceaxis = self.timeaxis
        else:
            self.sliceaxis = sliceaxis

#-----------------------------------------------------------------------------

def parse_pwout(fn_out, nl_dct=None, atspec=None, atpos_in=None):
    """
    args:
    -----
    fn_out : filename of the pw.x output file
    nl_cdt : dict returned by conf_namelists()
    atspec : dict returned by atomic_species()
    atpos_in : dict returned by atomic_positions()
    """
    verbose("[parse_pwout] parsing %s" %(fn_out))
    
    nstep = int(nl_dct['control']['nstep'])
    # Start temperature of MD run. Can also grep it from .out file, pattern for
    # re.search() (untested):
    # r'Starting temperature[ ]+=[ ]+([0-9\.])+[ ]+K'. Comes before the first 
    # 'ATOMIC_POSITIONS' and belongs to Rold.
    tempw = _float(nl_dct['ions']['tempw'])
    
    # Rold: (natoms x 3)
    Rold = atpos_in['R0']
    natoms = atpos_in['natoms']
    
##    DBG.t('parse-output')
    
    # Allocate R to store atomic coords.
    #
    # i = atom index: 0 ... natoms-1
    # j = time index: 0 ... nstep (NOTE: R.shape[1] = nstep+1 b/c j=0 -> Rold)
    # k = velocity or component index: x -> k=0, y -> k=1, z -> k=3
    # R[:,j,:] = (natoms x 3) array, atomic positions
    #
    Rshape = (natoms, nstep+1, 3)
    R = np.empty(Rshape, dtype=float)
    R[:,0,:] = Rold
    
    # temperature array
    T = np.empty((nstep+1,), dtype=float)
    T[0] = tempw
    
    # pressure array
    P = np.empty((nstep+1,), dtype=float)

    fh = open(fn_out)
    # R[:,0,:] = Rold, fill R[:,1:,:]
    j=1
    scan_atpos_rex = re.compile(r'^ATOMIC_POSITIONS[ ]*')
    scan_temp_rex = re.compile(r'[ ]+temperature[ ]+=[ ]+([0-9\.]+)[ ]+K')
    scan_stress_rex = re.compile(r'[ ]+total[ ]+stress[ ]+.*P.*=[ ]*(-*[0-9\.]+)')
    while True:
        
        # --- stress -----------------

        # Stress information for the *previous*, i.e. (j-1)th, iteration. P[0]
        # is the starting stress before the 1st MD iter. We do it this way b/c
        # we can't assign P[0] = p0 before the loop b/c we simply just don't 
        # know p0 from nowhere but the outfile.
        fh, flag, match = scan_until_pat2(fh, scan_stress_rex, err=False,
                                          retmatch=True)
        if flag == 0:
            verbose("[parse_pwout] stress scan: end of file "
                "'%s'" %fh.name)
            break
        else:            
            P[j-1] = _float(match.group(1))
        
        # --- ATOMIC_POSITIONS --------
        
        fh, flag = scan_until_pat2(fh, scan_atpos_rex, err=False)
        if flag == 0:
            verbose("[parse_pwout] atomic positions scan: end of file "
                "'%s'" %fh.name)
            break
        else:            
            # Rw: no copy, pointer to work array (view of slice), in-place
            # modification in function atomic_positions_out*()
            Rw = R[:,j,:]
    ##        fh = atomic_positions_out(fh, atpos_rex, Rw)
            fh = atomic_positions_out2(fh, natoms, Rw)
        
        # --- temperature -------------
        
        # usually, temperature appears after ATOMIC_POSITIONS
        fh, flag, match = scan_until_pat2(fh, scan_temp_rex, err=False,
                                          retmatch=True)
        if flag == 0:
            verbose("[parse_pwout] temperature scan: end of file "
                "'%s'" %fh.name)
            break
        else:            
            T[j] = _float(match.group(1))
        
        # --- method 1: compute velocities here ------------------------------
        # 
        # For this method, this function must return V instead of R. Not
        # used ATM. If we really can't afford the temporary array made in
        # velocity(), i.e. R is really, really big, then we should allocate R
        # in F-order, use f2py and do the loop over R[:,j,:] in Fortran.
        # Allocate R in F-order, b/c f2py will make a copy anyway if R is
        # C-cont.
        # 
        # plus:  small 2D temp arrays Rtmp
        # minus: no R array after loop for easy dumping
        # 
        # Note: If we decide to use method 1, then we could:
        #   - R shape: (natoms, nstep, 3), NOT (natoms, nstep+1, 3)
        #   - loop starts at j=0, not j=1
        #   - NO R[:,0,:] = Rold necessary
        #
        # uncomment + comment method 2 to use this! This leaves R shape at
        # (natoms, nstep+1, 3) and R[:,0,:] is not used.
        #
        ## Rtmp = Rw.copy()
        ## Rw -= Rold
        ## Rold = Rtmp
        ## # velocity
        ## Rw /= dt
        j += 1
    endj = j-1
    if endj != nstep:
        verbose("WARNING: file '%s' seems to short" %fn_out)
        verbose("    nstep = %s" %nstep)
        verbose("    iters in file = %s" %endj)
        verbose("    rest of output arrays (R, T, P) and all arrays depending "
              "on them will be zero or numpy.empty()")
    fh.close()     
##    DBG.pt('parse-output')
    return {'R': R, 'T': T, 'P': P, 'skipend': nstep-endj}

#-----------------------------------------------------------------------------
# computational
#-----------------------------------------------------------------------------

# "!>>>" in the docstring so that it won't get picked up by doctest, which
# looks for ">>>".
def normalize(a):
    """Normalize array by it's max value. Works also for complex arrays.

    example:
    --------
    !>>> a=np.array([3+4j, 5+4j])
    !>>> a
    array([ 3.+4.j,  5.+4.j])
    !>>> a.max()
    (5.0+4.0j)
    !>>> a/a.max()
    array([ 0.75609756+0.19512195j,  1.00000000+0.j ])
    """
    return a / a.max()

#-----------------------------------------------------------------------------

def norm(a):
    """2-norm for vectors."""
    _assert(len(a.shape) == 1, "input must be 1d array")
    # math.sqrt is faster then np.sqrt for scalar args
    return math.sqrt(np.dot(a,a))

#-----------------------------------------------------------------------------

def velocity(R, dt=None, copy=True, rslice=slice(None)):
    """Compute V from R. Essentially, time-next-neighbor-differences are
    calculated.
        
    args:
    -----
    R : 3D array, shape (natoms, nstep+1, 3)
        atomic coords with initial coords (Rold) at R[:,0,:]
    dt: float
        time step
    copy : bool
        In-place modification of R to save memory and avoid array
        copies. Use only if you don't use R after calling this function.
    rslice : slice object
        a slice for the 2nd axis (time axis) of R  
    
    returns:            
    --------
    V : 3D array, shape (natoms, nstep - offset, 3)

    notes:
    ------
    Even with copy=False, a temporary copy of R in the calculation is
    unavoidable.
                        
    """
    # --- method 2: compute velocities after loop  ---------------------------
    #
    # plus:  easy to dump R array if needed
    # minus: R[:,1:,:] - R[:,:-1,:] may be a large 3D temp (if nstep very
    #        large) But so far: nstep = 12000 -> R is 8 MB or so -> OK
    # note: R[:,0,:] must be == Rold !!        
    if copy:
        tmp = R.copy()[:,rslice,:]
    else:
        # view into R
        tmp = R[:,rslice,:]
    tmp[:,1:,:] =  tmp[:,1:,:] - tmp[:,:-1,:]
    # (natoms, nstep, 3), view only, skip j=0 <=> Rold
    V = tmp[:,1:,:]
    verbose("[velocity] V.shape: %s" %repr(V.shape))
    if dt is not None:
        V /= dt
    return V

#-----------------------------------------------------------------------------

def acorr(v, method=6):
    """Normalized autocorrelation function (ACF) for 1d arrys: 
    c(t) = <v(0) v(t)> / <v(0)**2>. 
    The x-axis is the offset "t" (or "lag" in Digital Signal Processing lit.).

    Several Python and Fortran implememtations. The Python versions are mostly
    for reference. For large arrays, only the pure numpy and Fortran versions
    fast and useful.

    args:
    -----
    v : 1d array
    method : int
        1: Python loops
        2: Python loops, zero-padded
        3: method 1, numpy vectorized
        4: uses numpy.correlate()
        5: fft, Wiener-Khinchin Theorem
        6: Fortran version of 1
        7: Fortran version of 3
    
    returns:
    --------
    c : numpy 1d array
        c[0]  <=> lag = 0
        c[-1] <=> lag = len(v)
    
    notes:
    ------
    speed:
        methods 1 ...  are loosely ordered slow .. fast
    methods:
       All methods, besides the FFT, are "exact", they use variations of loops
       in the time domain, i.e. norm(acorr(v,1) - acorr(v,6)) = 0.0. 
       The FFT method introduces small numerical noise, norm(acorr(v,1) -
       acorr(v,4)) = 4e-16 or so.

    signature of the Fortran extension _flib.acorr
        acorr - Function signature:
          c = acorr(v,c,method,[nstep])
        Required arguments:
          v : input rank-1 array('d') with bounds (nstep)
          c : input rank-1 array('d') with bounds (nstep)
          method : input int
        Optional arguments:
          nstep := len(v) input int
        Return objects:
          c : rank-1 array('d') with bounds (nstep)
    
    refs:
    -----
    [1] Numerical Recipes in Fortran, 2nd ed., ch. 13.2
    [2] http://mathworld.wolfram.com/FourierTransform.html
    [3] http://mathworld.wolfram.com/Cross-CorrelationTheorem.html
    [4] http://mathworld.wolfram.com/Wiener-KhinchinTheorem.html
    [5] http://mathworld.wolfram.com/Autocorrelation.html
    """#%_flib.acorr.__doc__
    nstep = v.shape[0]
    c = np.zeros((nstep,), dtype=float)
    if method == 1:
        for t in xrange(nstep):    
            for j in xrange(nstep-t):
                c[t] += v[j]*v[j+t] 
    elif method == 2:
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        for t in xrange(nstep):    
            for j in xrange(nstep):
                c[t] += v[j]*vv[j+t] 
    elif method == 3: 
        for t in xrange(nstep):
            c[t] = (v[:(nstep-t)] * v[t:]).sum()
    elif method == 4: 
        c = np.correlate(v, v, mode='full')[nstep-1:]
    elif method == 5: 
##        verbose("doing import: from scipy import fft, ifft")
##        from scipy import fft, ifft
        # Correlation via fft. After ifft, the imaginary part is (in theory) =
        # 0, in practise < 1e-16.
        # Cross-Correlation Theorem:
        #   corr(a,b)(t) = Int(-oo, +oo) a(tau)*conj(b)(tau+t) dtau   
        #                = ifft(fft(a)*fft(b).conj())
        # If a == b (like here), then this reduces to the special case of the 
        # Wiener-Khinchin Theorem (autocorrelation of `a`):
        #   corr(a,a) = ifft(np.abs(fft(a))**2)
        # Both theorems assume *periodic* data, i.e. `a` and `b` repeat after
        # `nstep` points. To deal with non-periodic data, we use zero-padding
        # at the end of `a` [1]. The result `c` contains the correlations for
        # positive and negative lags. Since in the ACF is symmetric around
        # lag=0, we return 0 ... +lag.
        vv = np.concatenate((v, np.zeros((nstep,),dtype=float)))
        c = ifft(np.abs(fft(vv))**2.0)[:nstep].real
    elif method == 6: 
        return _flib.acorr(v, c, 1)
    elif method == 7: 
        return _flib.acorr(v, c, 2)
    else:
        raise ValueError('unknown method: %s' %method)
    return c / c[0]

#-----------------------------------------------------------------------------

def pyvacf(V, m=None, method=3):
    """
    Reference implementation. We do some crazy numpy vectorization here
    for speedups.
    """
    natoms = V.shape[0]
    nstep = V.shape[1]
    c = np.zeros((nstep,), dtype=float)
    # we add extra multiplications by unity if m is None, but since it's only
    # the ref impl. .. who cares. better than having tons of if's in the loops.
    if m is None:
        m = np.ones((natoms,), dtype=float)

    if method == 1:
        # c(t) = <v(t0) v(t0 + t)> / <v(t0)**2> = C(t) / C(0)
        #
        # "displacements" `t'
        for t in xrange(nstep):
            # time origins t0 == j
            for j in xrange(nstep-t):
                for i in xrange(natoms):
                    c[t] += np.dot(V[i,j,:], V[i,j+t,:]) * m[i]
    elif method == 2:    
        # replace 1 inner loop
        for t in xrange(nstep):
            for j in xrange(nstep-t):
                # Multiply with mass-vector m:
                # Use array broadcasting: each col of V[:,j,:] is element-wise
                # multiplied with m (or in other words: multiply each of the 3
                # vectors V[:,j,k] in V[:,j,:] element-wise with m).
                #   V[:,j,:]          -> (natoms, 3)
                #   m[:,np.newaxis]    -> (natoms, 1)
                c[t] += (V[:,j,:] * V[:,j+t,:] * m[:,np.newaxis]).sum()
    elif method == 3:    
        # replace 2 inner loops, the last loop can't be vectorized (at least I
        # don't see how), `t' as array of indices doesn't work in numpy and
        # fortran
        
        # Multiply with mass-vector m:
        # method A: 
        # DON'T USE!! IT WILL OVERWRITE V !!!
        # Multiply whole V (each vector V[:,j,k]) with sqrt(m), m =
        # sqrt(m)*sqrt(m) in the dot product, then. 3x faster than method B. 
        #   m[:,np.newaxis,np.newaxis] -> (natoms, 1, 1)
        #   V                        -> (natoms, nstep, 3)
        ## V *= np.sqrt(m[:,np.newaxis,np.newaxis])
        for t in xrange(nstep):
            # method B: like in method 2, but extended to 3D
            c[t] = (V[:,:(nstep-t),:] * V[:,t:,:]*m[:,np.newaxis,np.newaxis]).sum()
##            c[t] = (V[:,:(nstep-t),:] * V[:,t:,:]).sum()
    else:
        raise ValueError('unknown method: %s' %method)
    # normalize to unity
    c = c / c[0]
    return c

#-----------------------------------------------------------------------------

def fvacf(V, m=None, method=2):
    """
    5+ times faster than pyvacf. Only 5 times b/c pyvacf is already
    partially numpy-optimized.

    notes:
    ------
    $ python -c "import _flib; print _flib.vacf.__doc__"
    vacf - Function signature:
      c = vacf(v,m,c,method,use_m,[natoms,nstep])
    Required arguments:
      v : input rank-3 array('d') with bounds (natoms,nstep,3)
      m : input rank-1 array('d') with bounds (natoms)
      c : input rank-1 array('d') with bounds (nstep)
      method : input int
      use_m : input int
    Optional arguments:
      natoms := shape(v,0) input int
      nstep := shape(v,1) input int
    Return objects:
      c : rank-1 array('d') with bounds (nstep)
    """
    natoms = V.shape[0]
    nstep = V.shape[1]
    # `c` as "intent(in, out)" could be "intent(out), allocatable" or so,
    # makes extension more pythonic, don't pass `c` in, let be allocated on
    # Fortran side
    c = np.zeros((nstep,), dtype=float)
    if m is None:
        # dummy
        m = np.empty((natoms,), dtype=float)
        use_m = 0
    else:
        use_m = 1
    # With "order='F'", we convert V to F-order and a copy is made. If we don't
    # do it, the wrapper code does. This copy is unavoidable, unless we
    # allocate the array in F order in the first place.
##    c = _flib.vacf(np.array(V, order='F'), m, c, method, use_m)
    c = _flib.vacf(V, m, c, method, use_m)
    return c

# alias
vacf = fvacf

#-----------------------------------------------------------------------------

def norm_int(y, x, val=1.0):
    """Normalize integral area of y(x) to `val`.
    
    args:
    -----
    x,y : numpy 1d arrays
    val : float

    returns:
    --------
    scaled y

    notes:
    ------
    The argument order y,x might be confusing. x,y would be more natural but we
    stick to order used in the scipy.integrate routines.
    """
#>>>>>>>>>>>>>>>>>>>>>>>    
    # First, scale x and y to the same order of magnitude before integration.
    # Not sure if this is necessary.
    fx = 1.0 / np.abs(x).max()
    fy = 1.0 / np.abs(y).max()
    # scaled vals
    sx = fx*x
    sy = fy*y
#-------------------
##    fx = fy = 1.0
##    sx=x
##    sy=y
#<<<<<<<<<<<<<<<<<<<<<<    
    # Area under unscaled y(x).
    area = simps(sy, sx) / (fx*fy)
    return y*val/area

#-----------------------------------------------------------------------------

def direct_pdos(V, dt=1.0, m=None, full_out=False):
    massvec=m 
    time_axis=1
    # array of V.shape, axis=1 is the fft of the arrays along axis 1 of V
    fftv = np.abs(fft(V, axis=time_axis))**2.0
    if massvec is not None:
        _assert(len(massvec) == V.shape[0], "len(massvec) != V.shape[0]")
        fftv *= massvec[:,np.newaxis, np.newaxis]
    # average remaining axes        
    full_pdos = fftv.sum(axis=0).sum(axis=1)        
    full_faxis = np.fft.fftfreq(V.shape[time_axis], dt)
    split_idx = len(full_faxis)/2
    faxis = full_faxis[:split_idx]
    pdos = full_pdos[:split_idx]
    
    default_out = (faxis, norm_int(pdos, faxis))
    extra_out = (full_faxis, full_pdos, split_idx)
    if full_out:
        return default_out + (extra_out,)
    else:
        return default_out

#-----------------------------------------------------------------------------

def vacf_pdos(V, dt=1.0, m=None, mirr=False, full_out=False):
    massvec=m 
    c = vacf(V, m=massvec)
    if mirr:
        verbose("[vacf_pdos] mirror VACF at t=0")
        cc = mirror(c)
    else:
        cc = c
       
    fftcc = fft(cc)
    # in Hz
    full_faxis = np.fft.fftfreq(fftcc.shape[0], dt)
    full_pdos = np.abs(fftcc)
    split_idx = len(full_faxis)/2
    faxis = full_faxis[:split_idx]
    pdos = full_pdos[:split_idx]

    default_out = (faxis, norm_int(pdos, faxis))
    extra_out = (full_faxis, fftcc, split_idx, c)
    if full_out:
        return default_out + (extra_out,)
    else:
        return default_out


#-----------------------------------------------------------------------------

def dft(a, method='loop'):
    """Simple straightforward complex DFT algo.
    
    args:
    -----
    a : numpy 1d array
    method : string, {'matmul', 'loop'}
    
    returns: 
    --------
    (len(a),) array

    examples:
    ---------
    >>> from scipy.fftpack import fft
    >>> a=np.random.rand(100)
    >>> sfft=fft(a)
    >>> dfft1=dft(a, method='loop')
    >>> dfft2=dft(a, method='matmul')
    >>> np.testing.assert_array_almost_equal(sfft, dfft1)
    >>> np.testing.assert_array_almost_equal(sfft, dfft2)

    notes:
    ------
    This is only a reference implementation and has it's limitations.
        'loop': runs looong
        'matmul': memory limit
        => use only with medium size arrays

    N = len(a)
    sqrt(-1) == np.sqrt(1.0 + 0.0*j) = 1.0j

    Forward DFT, see [2,3], scipy.fftpack.fft():
        y[k] = sum(n=0...N-1) a[n] * exp(-2*pi*n*k*sqrt(-1)/N)
        k = 0 ... N-1
    
    Backward DFT, see [1] eq. 12.1.6, 12.2.2:
        y[k] = sum(n=0...N-1) a[n] * exp(2*pi*n*k*sqrt(-1)/N)
        k = 0 ... N-1

    The algo for method=='matmul' is the matrix mult from [1], but as Forward
    DFT for comparison with scipy. The difference between FW and BW DFT is that
    the imaginary parts are mirrored around y=0. 

    [1] Numerical Recipes in Fortran, Second Edition, 1992
    [2] http://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
    [3] http://mathworld.wolfram.com/FourierTransform.html
    """
    pi = constants.pi
    N = a.shape[0]
    # n and k run from 0 ... N-1
    nk = np.linspace(0.0, float(N), endpoint=False, num=N)

    if method == 'loop':
        fta = np.empty((N,), dtype=complex)
        for k in nk:
            fta[k] = np.sum(a*np.exp(-2.0*pi*1.0j*k*nk/float(N)))
    elif method == 'matmul':
        # `mat` is the matrix with elements W**(n*k) in [1], eq. 12.2.2
        nkmat = nk*nk[:,np.newaxis]
        mat = np.exp(-2.0*pi*1.0j*nkmat/float(N))
        fta = np.dot(mat, a)
    else:
        raise ValueError("illegal method '%s'" %method)
    return fta            


#-----------------------------------------------------------------------------

def dft_axis(arr, axis=-1):
    """Same as scipy.fftpack.fft(arr, axis=axis), but *much* slower."""
    return np.apply_along_axis(dft, axis, arr)
 

#-----------------------------------------------------------------------------

def mirror(a):
    # len(aa) = 2*len(a) - 1
    return np.concatenate((a[::-1],a[1:]))

#-----------------------------------------------------------------------------

def slicetake(a, sl, axis=None, copy=False):
    """The equivalent of numpy.take(a, ..., axis=<axis>), but accepts slice
    objects instead of an index array. Also by default, it returns a *view* and
    no copy.
    
    args:
    -----
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list of tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : return a copy instead of a view
    
    returns:
    --------
    A view into `a`.

    examples:
    ---------
    >>> from numpy import s_
    >>> a = np.random.rand(20,20,20)
    >>> b1 = a[:,:,10:]
    >>> # single slice for axis 2 
    >>> b2 = slicetake(a, s_[10:], axis=2)
    >>> # tuple of slice objects 
    >>> b3 = slicetake(a, s_[:,:,10:])
    >>> (b2 == b1).all()
    True
    >>> (b3 == b1).all()
    True

    notes:
    ------
    
    1) Why do we need that:
    
    # no problem
    a[5:10:2]
    
    # the same, more general
    sl = slice(5,10,2)
    a[sl]

    But we want to:
     - Define (type in) a slice object only once.
     - Take the slice of different arrays along different axes.
    Since numpy.take() and a.take() don't handle slice objects, one would have
    to use direct slicing and pay attention to the shape of the array:
          a[sl], b[:,:,sl,:], etc ... not practical.
    
    Example of things that don't work with numpy-only tools:

        R = R[:,sl,:]
        T = T[sl]
        P = P[sl]
    
    This is not generic. We want to use an 'axis' keyword instead. np.r_()
    generates index arrays from slice objects (e.g r_[1:5] == r_[s_[1:5]
    ==r_[slice(1,5,None)]). Since we need index arrays for numpy.take(), maybe
    we can use that?
        
        R = R.take(np.r_[sl], axis=1)
        T = T.take(np.r_[sl], axis=0)
        P = P.take(np.r_[sl], axis=0)
    
    But it does not work alawys:         
    np.r_[slice(...)] does not work for all slice types. E.g. not for
        
        r_[s_[::5]] == r_[slice(None, None, 5)] == array([], dtype=int32)
        r_[::5]                                 == array([], dtype=int32)
        r_[s_[1:]]  == r_[slice(1, None, None)] == array([0])
        r_[1:]
            ValueError: dimensions too large.
    
    The returned index arrays are wrong (or we even get an exception).
    The reason is that s_ translates a fancy index (1:, ::5, 1:10:2,
    ...) to a slice object. This always works. But since take()
    accepts only index arrays, we use r_[s_[<fancy_index>]], where r_
    translates the slice object prodced by s_ to an index array.
    THAT works only if start and stop of the slice are known. r_ has
    no way of knowing the dimensions of the array to be sliced and so
    it can't transform a slice object into a correct index arry in case
    of slice(<number>, None, None) or slice(None, None, <number>).

    2) Slice vs. copy
    
    numpy.take(a, array([0,1,2,3])) or a[array([0,1,2,3])] return a copy of `a` b/c
    that's "fancy indexing". But a[slice(0,4,None)], which is the same as
    indexing (slicing) a[:4], return *views*. 
    """
    # numpy.s_[:] == slice(0, None, None). The same works with  
    # slice(None) == slice(None, None, None).
    if axis is None:
        slices = sl
    else:        
        slices = [slice(None)]*len(a.shape)
        slices[axis] = sl
    # a[...] can take a tuple or list of slice objects
    # a[x:y:z, i:j:k] is the same as
    # a[(slice(x,y,z), slice(i,j,k))] == a[[slice(x,y,z), slice(i,j,k)]]
##    return a[tuple(slices)]
    if copy:
        return a[slices].copy()
    else:        
        return a[slices]

#-----------------------------------------------------------------------------

def sliceput(a, b, sl, axis=None):
    """The equivalent of a[<slice or index>]=b, but accepts slices objects
    instead of array indices (e.g. a[:,1:]).
    
    args:
    -----
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list of tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    
    returns:
    --------
    The modified `a`.
    
    examples:
    ---------
    >>> from numpy import s_
    >>> a=np.arange(12).reshape((2,6))
    >>> a[:,1:3] = 100
    >>> a
    array([[  0, 100, 100,   3,   4,   5],
           [  6, 100, 100,   9,  10,  11]])
    >>> sliceput(a, 200, s_[1:3], axis=1)
    array([[  0, 200, 200,   3,   4,   5],
           [  6, 200, 200,   9,  10,  11]])
    >>> sliceput(a, 300, s_[:,1:3])
    array([[  0, 300, 300,   3,   4,   5],
           [  6, 300, 300,   9,  10,  11]])
    """
    if axis is None:
        # silce(...) or (slice(...), slice(...), ...)
        tmp = sl
    else:
        # [slice(...), slice(...), ...]
        tmp = [slice(None)]*len(a.shape)
        tmp[axis] = sl
##    a[tuple(tmp)] = b
    a[tmp] = b
    return a

#-----------------------------------------------------------------------------
# crystal structure
#-----------------------------------------------------------------------------

def scell_mask(dim1, dim2, dim3):
    """Build a mask for the creation of a dim1 x dim2 x dim3 supercell (for 3d
    coordinates).  Return all possible permutations with repitition of the
    integers n1, n2,  n3, and n1, n2, n3 = 0, ..., dim1-1, dim2-1, dim3-1 .

    args:
    -----
    dim1, dim2, dim3 : int

    returns:
    --------
    mask : 2d array, shape (dim1*dim2*dim3, 3)

    example:
    --------
    >>> # 2x2x2 supercell
    >>> scell_mask(2,2,2)
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  1.,  1.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  1.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  1.]])
    >>> # a "plane" of 4 cells           
    >>> scell_mask(2,2,1)
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.]])
    
    notes:
    ------
    If dim1 == dim2 == dim3 == n, then we have a permutation with repetition
    (german: Variation mit Wiederholung):  select r elements out of n with
    rep. In gerneral, n >= r or n < r possible. There are always n**r
    possibilities.
    Here r = 3 always (select x,y,z direction):
    example:
    n = 2 <=> 2x2x2 supercell: 
      all 3-tuples out of [0,1]   -> n**r = 2**3 = 8
    n=3 <=> 3x3x3 supercell:
      all 3-tuples out of [0,1,2] -> n**r = 3**3 = 27
    Computationally, we need `r` nested loops, one per dim.  
    """
    b = [] 
    for n1 in xrange(dim1):
        for n2 in xrange(dim2):
            for n3 in xrange(dim3):
                b.append([n1,n2,n3])
    return np.array(b, dtype=float)

#-----------------------------------------------------------------------------

def scell(R0, cp, mask, symbols):
    """Build supercell based on `mask`.

    args:
    -----
    R0 : 2d array, (natoms, 3) with atomic coords
    cp : 2d array, (3, 3)
        cell parameters, primitive lattice vecs as *rows* (see
        cell_parameters())
    mask : what scell_mask() returns, (N, 3)
    symbols : list of strings with atom symbols, (natoms,)

    returns:
    --------
    (sc_symbols, Rsc)
    Rsc : array (N*natoms, 3)
    sc_symbols : list of strings with atom symbols, (N*natoms,)

    notes:
    ------
    `R0` and `cp` must be in the same coordinate system and unit!! Here, `cp` is in
    cartesian coords and in alat or a.u. unit.
    """
    sc_symbols = []
    Rsc = np.empty((mask.shape[0]*R0.shape[0], 3), dtype=float)
    k = 0
    for i in xrange(R0.shape[0]):
        for j in xrange(mask.shape[0]):
            # Build supercell. Place each atom N=dim1*dim2*dim3 times in the
            # supercell, i.e. copy unit cell N times. Actually, N-1, since
            # n1=n2=n3=0 is the unit cell itself.
            # mask[j,:] = [n1, n2, n3], ni = integers (floats actually, but
            #   floor(ni) == ni)
            # cp = [[-- a1 --]
            #       [-- a2 --]
            #       [-- a3 --]]
            # dot(...) = n1*a1 + n2*a2 + n3*a3
            # R0[i,:] = r_i = position vect of atom i in the unit cell
            # r_i_in_supercell = r_i + n1*a1 + n2*a2 + n3*a3
            #   for all permutations (see scell_mask()) of n1, n2, n3.
            #   ni = 0, ..., dimi-1
            Rsc[k,:] = R0[i,:] + np.dot(mask[j,:], cp)
            sc_symbols.append(symbols[i])
            k += 1
    return sc_symbols, Rsc
            
#-----------------------------------------------------------------------------

def coord_trans(R, old=None, new=None, copy=True, align='cols'):
    """Coordinate transformation.
    
    args:
    -----
    R : array with coordinates in old coord sys `old`. 
        The last dim must be the number of coordinates, i.e. R.shape[-1] == 3
        for normal 3-dim x,y,z.
    old, new : 2d arrays
        matrices with the old and new basis vectors as rows or cols
    copy : bool, optional
        True: overwrite `R`
        False: return new array
    align : string
        {'cols', 'rows'}
        cols : basis vecs are columns of `old` and `new`
        rows : basis vecs are rows    of `old` and `new`

    returns:
    --------
    array of shape = R.shape, coordinates in system `new`
    
    examples:
    ---------
    # Taken from [1].
    >>> import numpy as np
    >>> import math
    >>> v = np.array([1.0,1.5])
    >>> I = np.identity(2)
    >>> X = math.sqrt(2)/2.0*np.array([[1,-1],[1,1]])
    >>> Y = np.array([[1,1],[0,1]])
    >>> coord_trans(v,I,I)
    array([ 1. ,  1.5])
    >>> v_X = coord_trans(v,I,X)
    >>> v_Y = coord_trans(v,I,Y)
    >>> v_X
    array([ 1.76776695,  0.35355339])
    >>> v_Y
    array([-0.5,  1.5])
    >>> coord_trans(v_Y,Y,I)
    array([ 1. ,  1.5])
    >>> coord_trans(v_X,X,I)
    array([ 1. ,  1.5])
    
    >>> Rold = np.random.rand(30,200,3)
    >>> old = np.random.rand(3,3)
    >>> new = np.random.rand(3,3)
    >>> Rnew = coord_trans(Rold, old=old, new=new)
    >>> Rold2 = coord_trans(Rnew, old=new, new=old)
    >>> np.testing.assert_almost_equal(Rold, Rold2)
    
    # these do the same: A, B have vecs as rows
    >>> RB1=coord_trans(Rold, old=old, new=new, align='rows') 
    >>> RB2=coord_trans(Rold, old=old.T, new=new.T) 
    >>> np.testing.assert_almost_equal(Rold, Rold2)

    refs:
    [1] http://www.mathe.tu-freiberg.de/~eiermann/Vorlesungen/HM/index_HM2.htm
        Kapitel 6
    
    notes:
    ------
    Coordinate transformation:
        
        Mathematical formulation:
        X, Y square matrices with basis vecs as *columns*.

        X ... old, shape: (3,3)
        Y ... new, shape: (3,3)
        I ... identity matrix, basis vecs of cartesian system, shape: (3,3)
        A ... transformation matrix, shape(3,3)
        v_X ... column vector v in basis X, shape: (3,1)
        v_Y ... column vector v in basis Y, shape: (3,1)
        v_I ... column vector v in basis I, shape: (3,1)

        "." denotes matrix multiplication:
        
        Y . v_Y = X . v_X = I . v_I = v_I
        v_Y = Y^-1 . X . v_X == A . v_X
        v_Y^T = (A . v_X)^T  # in general: (A . B)^T = B^T . A^T 
              = v_X^T . A^T
        
        Numpy:

        In numpy, v^T == v, and so no "vector" needs to be transposed:
            v_Y^T = v_X^T . A^T 
        becomes
            v_Y = v_X . A^T 
        and that is the form implemented here.            
        That's because a vector is a 1d array, e.g. v = array([1,2,3]) with
        shape (3,) and rank 1 instead of column or row vector ((3,1) or (1,3))
        and rank 2. Transposing is not defined: v.T == v .  The dot() function
        knows that and performs the correct multiplication accordingly. 

        Example:
        
        Transformation from crystal to cartesian coords.

        old:
        X = coord sys for a hexagonal lattice with primitive lattice
            vectors (basis vectors) a0, a1, a2, each shape (3,)
        new:                
        Y = cartesian, i.e. the components a0[i], a1[i], a2[i] of the 
            crystal basis vectors are cartesian:
                a0 = a0[0]*[1,0,0] + a0[1]*[0,1,0] + a0[2]*[0,0,1]
                a1 = a1[0]*[1,0,0] + a1[1]*[0,1,0] + a1[2]*[0,0,1]
                a2 = a2[0]*[1,0,0] + a2[1]*[0,1,0] + a2[2]*[0,0,1]
        v = shape (3,) vec in the hexagonal lattice ("crystal
            coordinates")
        
        We have 
            
            A = (Y^-1 . X) = X
        
        since Y == I and I^-1 == I.

            v_Y = v_I = X . v_X = A . v_X
        
        Let the a's be the *rows* of the transformation matrix. In general,
        it's more practical to use dot(v,A.T) instead of dot(A,v). See below.
            
            A^T == A.T = [[--- a0 ---], 
                          [--- a1 ---], 
                          [--- a2 ---]] 
        and let
            
            v == v_X. 
        
        Every product X . v_X is actually an expansion of v_X in the basis
        vectors contained in X.
        We expand v_X in terms of the crystal basis (X):         
            
            v_Y = 
              = v[0]*a0       + v[1]*a1       + v[2]*a2
              
              = v[0]*A.T[0,:] + v[1]*A.T[1,:] + v[2]*A.T[2,:]
              
              = [v[0]*A.T[0,0] + v[1]*A.T[1,0] + v[2]*A.T[2,0],
                 v[0]*A.T[0,1] + v[1]*A.T[1,1] + v[2]*A.T[2,1],
                 v[0]*A.T[0,2] + v[1]*A.T[1,2] + v[2]*A.T[2,2]]
              
              = dot(v, A.T)       <=> v[j] = sum(i=0..2) v[i]*A[i,j]
              = dot(A, v)         <=> v[i] = sum(j=0..2) A[i,j]*v[j]
         
        If this dot product is actually computed, we get v in cartesian coords
        (Y).  
    
    shape of `R`:
        
        If we want to use fast numpy array broadcasting to transform many `v`
        vectors at once, we must use the form dot(R,A.T).
        The shape of `R` doesn't matter, as long as the last dimension matches
        the dimensions of A (e.g. R: (n,m,3), A: (3,3), dot(R,A.T): (n,m,3)).
         
        1d: R.shape = (3,)
        R == v = [x,y,z] 
        -> dot(A, v) == dot(v,A.T) = [x', y', z']

        2d: R.shape = (N,3)
        Array of coords of N atoms, R[i,:] = coord of i-th atom. The dot
        product is broadcast along the first axis of R (i.e. *each* row of R is
        dot()'ed with A.T).
        R = 
        [[x0,       y0,     z0],
         [x1,       y1,     z1],
          ...
         [x(N-1),   y(N-1), z(N-1)]]
        -> dot(R,A.T) = 
        [[x0',     y0',     z0'],
         [x1',     y1',     z1'],
          ...
         [x(N-1)', y(N-1)', z(N-1)']]
        
        3d: R.shape = (natoms, nstep, 3) 
        R[i,j,:] is the shape (3,) vec of coords for atom i at time step j.
        Broadcasting along the first and second axis. 
        These loops have the same result as newR=dot(R, A.T):
            # New coords in each (nstep, 3) matrix R[i,...] containing coords
            # of atom i for each time step. Again, each row is dot()'ed.
            for i in xrange(R.shape[0]):
                newR[i,...] = dot(R[i,...],A.T)
            
            # same as example with 2d array: R[:,j,:] is a matrix with atom
            # coord on each row at time step j
            for j in xrange(R.shape[1]):
                newR[:,j,:] = dot(R[:,j,:],A.T)
    
    """
    _assert(old.shape[0] == new.shape[0], "dim 0 of `old` and `new` "
        "doenn't match")
    _assert(old.shape[1] == new.shape[1], "dim 1 of `old` and `new` "
        "doenn't match")
    msg = ''        
    if align == 'rows':
        old = old.T
        new = new.T
        msg = 'after transpose, '
    _assert(R.shape[-1] == old.shape[0], "%slast dim of `R` must match first dim"
        " of `old` and `new`" %msg)
    if copy:
        tmp = R.copy()
    else:
        tmp = R
    # move import to module level if needed regularly
##    verbose("[coord_trans] doing import: from scipy.linalg import inv") 
##    from scipy.linalg import inv
    # must use `tmp[:] = ...`, just `tmp = ...` is a new array
    tmp[:] = np.dot(tmp, np.dot(inv(new), old).T)
    return tmp
        
#-----------------------------------------------------------------------------
# misc
#-----------------------------------------------------------------------------

def verbose(msg):
    if VERBOSE:
        print(msg)

#-----------------------------------------------------------------------------

def str_arr(a):
    return str(a).replace('[', '').replace(']', '')

#-----------------------------------------------------------------------------

def system(call):
    """
    Primitive os.system() replacement. stdout and stderr go to the shell. Only
    diff: Waits until child process is complete. 

    args:
    ----
    call : string (example: 'ls -l')
    """
    p = S.Popen(call, shell=True)
    os.waitpid(p.pid, 0)

#-----------------------------------------------------------------------------

def fullpath(s):
    """Complete path: absolute path + $HOME expansion."""
    return os.path.abspath(os.path.expanduser(s))

#-----------------------------------------------------------------------------

def _test_atpos(R, pw_fn='SiAlON.example_md.out'):
    """
    Use this script and perl, parse THE SAME `pw_fn`, write the result to
    files, load them in IPython and compare them.
    
    args:
    -----
    R : 3D array of read atomic coords,  shape (natoms, nstep, 3)
    """
    DBG.t('_test_atpos')
    os.unlink('out.python')
    os.unlink('out.perl')
    verbose("perl extract + write ...")
    system("perl -ne 'print \"$2\n\" if /(Al|O|Si|N)(([ ]+-*[0-9]+\.*[0-9]*){3})/'"
           " < %s > out.perl" %pw_fn)
    verbose("python write ...")
    fhout = open('out.python', 'a')
    # loop over nstep
    for j in xrange(R.shape[1]):
        np.savetxt(fhout, R[:,j,:])
    verbose((R == 0.0).any())
    verbose(R.shape)
    fhout.close()
    DBG.pt('_test_atpos')

#-----------------------------------------------------------------------------


def main(opts):
    """Main function.

    args:
    ----
    opts: output of `opts=parser.parse_args()`, where
        `parser=argparse.ArgumentParser()`
    """

    # print options
    verbose("options:")
    # opts.__dict__: a dict with options and values {'opt1': val1, 'opt2':
    # val2, ...}
    for key, val in opts.__dict__.iteritems():
        verbose("    %s: %s" %(key, repr(val)))
    
    # make outdir
    if not os.path.exists(opts.outdir):
        verbose("creating outdir: %s" %opts.outdir)
        os.mkdir(opts.outdir)
    
    # make filenames
    if opts.pwofn.endswith('.out'):
        fn_body = os.path.basename(opts.pwofn.replace('.out', ''))
    else:
        fn_body = os.path.basename(opts.pwofn)
    if opts.file_type == 'bin':
        file_suffix = '.dat'
    elif opts.file_type == 'txt':
        file_suffix = '.txt'
    else:
        raise StandardError("wrong opts.file_type")
    verbose("fn_body: %s" %fn_body        )
    vfn = pjoin(opts.outdir, fn_body + '.v' + file_suffix)
    rfn = pjoin(opts.outdir, fn_body + '.r' + file_suffix)
    mfn = pjoin(opts.outdir, fn_body + '.m' + file_suffix)
    tfn = pjoin(opts.outdir, fn_body + '.temp' + file_suffix)
    vacffn = pjoin(opts.outdir, fn_body + '.vacf' + file_suffix)
    pfn = pjoin(opts.outdir, fn_body + '.p' + file_suffix)
    pdosfn = pjoin(opts.outdir, fn_body + '.pdos' + file_suffix)
    
    # needed in 'parse' and 'dos'
    nl_dct = conf_namelists(opts.pwifn)
    
    # --- parse and write ----------------------------------------------------
    
    if opts.parse:
        atspec = atomic_species(opts.pwifn)
        atpos_in = atomic_positions(opts.pwifn, atspec)
        
        # This is a bit messy: call every single parse function (atomic_*(),
        # conf_namelists()) and pass their output as args to parse_pwout(). Why
        # not make a big function that does all that under the hood? Because we
        # plan to use a class Parser one day which will have all these
        # individual functions as methods. Individual output args will be
        # shared via data members. Stay tuned.
        pwout = parse_pwout(fn_out=opts.pwofn,
                            nl_dct=nl_dct, atspec=atspec,
                            atpos_in=atpos_in)
        
        massvec = atpos_in['massvec']
        Rfull = pwout['R']
        Tfull = pwout['T']
        Pfull = pwout['P']
         
        # Handle outfile from aborted/killed calcs. 
        #
        # We consider a "full" data array an array consisting of only
        # meaningful data, i.e. not np.zeros() or np.empty(). That's why we
        # cut off the useless end parts of the arrays and "overwrite" the 
        # "full" arrays here.
        if opts.skipend and (pwout['skipend'] > 0):
            # XXX If used in Parser class, use slicetake() or
            # something, i.e. use axis information, not direct indexing.
            Rfull = Rfull[:,:-pwout['skipend'],:]
            Tfull = Tfull[:-pwout['skipend']]
            Pfull = Pfull[:-pwout['skipend']]
            verbose("skipping %s time points at end of R, new shape: %s" 
                  %(pwout['skipend'], str(Rfull.shape)))
            verbose("skipping %s time points at end of T, new shape: %s"
                  %(pwout['skipend'], str(Tfull.shape)))
            verbose("skipping %s time points at end of P, new shape: %s"
                  %(pwout['skipend'], str(Pfull.shape)))
        verbose("mean temperature: %f" %Tfull.mean())
        verbose("mean pressure: %f" %Pfull.mean())
        
        # Slice arrays if -s option was used. This does not affect the "full"
        # arrays. Slices are only views.
        if opts.slice is not None:
            verbose("slicing arrays")
            verbose("    R: old shape: %s" %str(Rfull.shape))
            verbose("    T: old shape: %s" %str(Tfull.shape))
            verbose("    P: old shape: %s" %str(Pfull.shape))
            R = slicetake(Rfull, opts.slice, axis=1)
            T = slicetake(Tfull, opts.slice, axis=0)
            P = slicetake(Pfull, opts.slice, axis=0)
            verbose("    R: new shape: %s" %str(R.shape))
            verbose("    T: new shape: %s" %str(T.shape))
            verbose("    P: new shape: %s" %str(P.shape))
        else:
            R = Rfull
            T = Tfull
            P = Pfull
        
        # If 
        #   ibrav=0 
        #   CELL_PARAMETERS is present
        #   ATOMIC_POSITIONS crystal
        # and if requested by the user, we transform the atomic coords crystal
        # -> cartesian.
        #
        # If the pw.x calc went fine, then the VACF calculated with R in
        # cartesian alat|bohr|angstrom or crystal must be the same. The
        # coord trans here is actually not necessary at all and serves only for
        # testing/verification purposes.
        # 
        #
        # allowed ATOMIC_POSITIONS units (from the Pwscf help):
        #    alat    : atomic positions are in cartesian coordinates,
        #              in units of the lattice parameter "a" (default)
        #
        #    bohr    : atomic positions are in cartesian coordinate,
        #              in atomic units (i.e. Bohr)
        #
        #    angstrom: atomic positions are in cartesian coordinates,
        #              in Angstrom
        #
        #    crystal : atomic positions are in crystal coordinates, i.e.
        #              in relative coordinates of the primitive lattice vectors
        #              (see below)
        # 
        unit = atpos_in['unit']
        ibrav = int(nl_dct['system']['ibrav'])
        verbose("unit: '%s'" %unit)
        verbose("ibrav: %s" %ibrav)
        # ATOMIC_POSITIONS angstrom  -> cartesian angstrom
        # ATOMIC_POSITIONS bohr      -> cartesian bohr  [== a.u.] 
        # ATOMIC_POSITIONS           -> cartesian alat
        # ATOMIC_POSITIONS alat      -> cartesian alat
        # ATOMIC_POSITIONS crystal   -> crystal alat | a.u. 
        #
        # if celldm(1) present  -> CELL_PARAMETERS in alat -> crystal alat
        # if not                -> CELL_PARAMETERS in a.u. -> crystal a.u.
        # 
        if unit == 'crystal' and opts.coord_trans:
            if ibrav == 0:
                cp = cell_parameters(opts.pwifn)
                # CELL_PARAMETERS in alat
                if nl_dct['system'].has_key('celldm(1)'):
                    alat = _float(nl_dct['system']['celldm(1)'])
                    verbose("alat:", alat)
                    verbose("assuming CELL_PARAMETERS in alat")
                    new_unit = 'alat'
                # CELL_PARAMETERS in a.u.
                else:
                    verbose("celldm(1) not present" )
                    verbose("assuming CELL_PARAMETERS in Rydberg a.u.")
                    new_unit = 'a.u.'
                verbose("doing coord transformation: %s -> %s" %('crystal',
                    'cartesian' + new_unit))
                R = coord_trans(R, old=cp, new=np.identity(cp.shape[0]),
                                copy=False, align='rows')
            else:
                verbose("ibrav != 0, ignoring possibly available card "
                "CELL_PARAMETERS, no coord transformation") 

        # write R here if it's overwritten in velocity()
        writearr(mfn, massvec,  type=opts.file_type)
        writearr(tfn, Tfull,     type=opts.file_type)
        writearr(rfn, Rfull,     type=opts.file_type, axis=1)
        writearr(pfn, Pfull,     type=opts.file_type)
        
    # ---  read --------------------------------------------------------------
    
    if opts.dos and not opts.parse:
        # Reload the only data that we get from parsing the in- and outfile. We
        # could, in some cases, also reload `V`, `c` and so on, but it's easier
        # to re-compute them as long as the calculations run only seconds.
        R = readbin(rfn)
        massvec = readbin(mfn)
    
    # --- dos ----------------------------------------------------------------
    
    if opts.dos: 

        # If we compute the *normalized* VCAF, then dt is a factor ( <v_x(0)
        # v_x(t)> = 1/dt^2 <dx(0) dx(t)> ) which cancels in the normalization.
        # dt is not needed in the velocity calculation.
        V = velocity(R, copy=False)

        # in s
        dt = _float(nl_dct['control']['dt']) * constants.tryd
        verbose("dt: %s seconds" %dt)
        if (opts.slice is not None) and (opts.slice.step is not None):
            verbose("scaling dt with slice step: %s" %opts.slice.step)
            dt *= float(opts.slice.step)
            verbose("    scaled dt: %s seconds" %dt)
        
        if not opts.mass:
            massvec = None
        else:
            verbose("mass-weighting VACF")
        
        if opts.dos_method == 'vacf':
            faxis, pdos, extra = vacf_pdos(V, dt=dt, m=massvec,
                mirr=opts.mirror, full_out=True)
            full_faxis, fftcc, split_idx, vacf_data = extra                
            real_imag_ratio = norm(fftcc.real) / norm(fftcc.imag)
            verbose("fft: real/imag: %s" %real_imag_ratio)
            pdos_out = np.empty((split_idx, 4), dtype=float)
            pdos_comment = textwrap.dedent("""
            # PDOS by FFT of VACF
            # Integral normalized to 1.0: int(abs(fft(vacf)), f) = 1.0 
            # f [Hz]  abs(fft(vacf))  fft(vacf).real  fft(vacf).imag 
            """)
        elif opts.dos_method == 'direct':
            faxis, pdos, extra = direct_pdos(V, dt=dt, m=massvec,
                full_out=True)
            full_faxis, full_pdos, split_idx = extra                
            real_imag_ratio = None
            pdos_out = np.empty((split_idx, 2), dtype=float)
            pdos_comment = textwrap.dedent("""
            # Direct PDOS
            # Integral normalized to 1.0: int(pdos, f) = 1.0 
            # f [Hz]  pdos
            """)
            
        df1 = full_faxis[1]-full_faxis[0]
        df2 = 1.0/len(full_faxis)/dt
        # f axis in in order 1e12, so small `decimal` values are sufficient
        np.testing.assert_almost_equal(df1, df2, decimal=0)
        df = df1
        
        pdos_out[:,0] = faxis
        pdos_out[:,1] = pdos
        if opts.dos_method == 'vacf':
            pdos_out[:,2] = normalize(fftcc[:split_idx].real)
            pdos_out[:,3] = normalize(fftcc[:split_idx].imag)
        info = dict()
        info['nyquist_freq'] = 0.5/dt,
        info['freq_resolution'] = df,
        info['dt'] = dt,
        if real_imag_ratio is not None:
            info['real_imag_ratio'] = real_imag_ratio,
        pdos_info = {'fft': info}

        writearr(pdosfn, pdos_out, info=pdos_info, comment=pdos_comment, type=opts.file_type)
        writearr(vfn, V, axis=1, type=opts.file_type)
        if opts.dos_method == 'vacf':
            writearr(vacffn, vacf_data, type=opts.file_type)

#-----------------------------------------------------------------------------
# main
#-----------------------------------------------------------------------------

if __name__ == '__main__':
    
##    # Turn off text messages from individual functions. They confuse doctest.
##    VERBOSE = False
##    import doctest
##    doctest.testmod(verbose=True)
    
    VERBOSE = True
    
    # argparse's bool options:
    # We use argparse.py instead of Python's standard lib optparse, b/c only
    # argparse supports optional arguments. We use this for our bool valued
    # options.
    #
    # cases:
    #   no -d       -> default
    #      -d       -> const
    #      -d VAL   -> tobool(VAL)
    #
    # For bool opts, all our "const" values are True, i.e. if the option is
    # used w/o argument, then it is True.

    import argparse
    import textwrap
    epilog=textwrap.dedent("""
    bool options
    ------------
    Examples use the '-d' option.  All bool options have default values. If an
    option is NOT given, the default is used. If the option is used and NO
    argument is given (e.g. just "-d"), then the corresponding variable is
    True. With arguments, e.g. "-d VAL", VAL can be
        
        1, true,  on,  yes -> True
        0, false, off, no  -> False
     
    Only disadvantage: You cannot use
      -dpm 
    must be 
      -d -p -m
    """)    
    parser = argparse.ArgumentParser(epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        version="%(prog)s " + str(VERSION))
    # --- bool flags with optional args  ------------------
    parser.add_argument('-p', '--parse',
        nargs='?',
        metavar='bool',
        const=True,
        type=tobool,
        default=False,
        help="parse pw.x in- and outfile [%(default)s]",
        )
    parser.add_argument('-d', '--dos', 
        nargs='?',
        metavar='bool',
        const=True,
        type=tobool,
        default=False,
        help="calculate PDOS [%(default)s]",
        )
    parser.add_argument('-m', '--mirror', 
        nargs='?',
        metavar='bool',
        const=True,
        type=tobool,
        default=True,
        help="mirror VACF around t=0 before fft [%(default)s]",
        )
    parser.add_argument('-M', '--mass', 
        nargs='?',
        metavar='bool',
        const=True,
        type=tobool,
        default=True,
        help="use mass-weighted VACF [%(default)s]",
        )
    parser.add_argument('-c', '--coord-trans', 
        nargs='?',
        metavar='bool',
        const=True,
        type=tobool,
        default=False,
        help="""perform coordinate transformation if atomic coords are not  
            in cartesian alat coords [%(default)s]""",
        )
    parser.add_argument('-e', '--skipend', 
        nargs='?',
        metavar='bool',
        const=True,
        type=tobool,
        default=True,
        help="""if PWOFN contains < nstep time iterations (unfinished
            calculation), then skip the last points [%(default)s]""",
        )
    # --- opts with required args -------------------------
    parser.add_argument('-x', '--outdir',
        default=pjoin(fullpath(os.curdir), 'pdos'),
        type=fullpath,
        help="[%(default)s]",
        )
    parser.add_argument('-i', '--pwi',
        dest="pwifn",
        type=fullpath,
        help="pw.x input file",
        )
    parser.add_argument('-o', '--pwo',
        dest="pwofn",
        type=fullpath,
        help="pw.x output file",
        )
    parser.add_argument('-s', '--slice',
        default=None,
        type=toslice,
        help="""indexing for the time axis of R in the velocity
            calculation (remember that Python's indexing is 0-based),
            examples: '3:', '3:7', '3:7:2', 3::2', ':7:2', '::-1',
            see also: http://scipy.org/Tentative_NumPy_Tutorial, 
            section "Indexing, Slicing and Iterating"
            """,  
        )
    parser.add_argument('-f', '--file-type', 
        default='bin',
        help="""{'bin', 'txt'} write data files binary or ASCII text 
            [%(default)s]""",
        )
    parser.add_argument('-t', '--dos-method', 
        default='vacf',
        help="""{'vacf', 'direct'} method to calculate PDOS  [%(default)s]""",
        )

    opts = parser.parse_args()    
    sys.exit(main(opts))
