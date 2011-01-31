# common.py
#
# File operations, common system utils and other handy tools.

# Import std lib's signal module, not the one from pwtools. AFAIK, this is
# needed for Python 2.4 ... 2.6. See PEP 328.
from __future__ import absolute_import
import signal

import types
import os
import subprocess
import shutil
import re
import ConfigParser
import numpy as np

# slow import time
#
# TODO: maybe move functions which need scipy functionality to smth like
# numutils.py or whatever
from scipy.integrate import simps
from scipy.interpolate import splev, splrep
from pwtools.verbose import verbose


def assert_cond(cond, string=None):
    """Use this instead of `assert cond, string`. It's been said on
    numpy-discussions that the assert statement shouldn't be used to test user
    input in functions b/c with `python ... -O0` or __debug__ not beeing
    defined, the statement is not tested.
    
    args:
    -----
    cond : bool
        True : None is returned
        False : exception is raised
    string : str
    
    example:
    --------
    assert_cond(1==1, 'lala') -> ok
    assert_cond(1==2, 'lala') -> exception is raised
    """
    if not cond:
        raise AssertionError(string)


#-----------------------------------------------------------------------------
# Config file stuff
#-----------------------------------------------------------------------------

class PydosConfigParser(ConfigParser.SafeConfigParser):
    """All values passed as `arg` to self.set(self, section, option, arg) are
    converted to a string with frepr(). get*() methods are the usual ones
    provided by the base class ConfigParser.SafeConfigParser: get(), getint(),
    getfloat(), getboolean(). Option keys are case-sensitive.
    """
    # make keys case-sensitive
    ConfigParser.SafeConfigParser.optionxform = str
    def set(self, section, option, arg):
        ConfigParser.SafeConfigParser.set(self, section, option, frepr(arg))


def add_to_config(config, info):
    """Add sections and key-val paris in `info` to `config`.
    
    args:
    -----
    config : ConfigParser object
    info : dict of dicts, see io.writearr()

    returns:
    --------
    modified config
    """
    for sec, dct in info.iteritems():
        config.add_section(sec)
        for key, val in dct.iteritems():
            config.set(sec, key, val)
    return config


#-----------------------------------------------------------------------------
# Type converters / light numerical stuff
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
    assert_cond(isinstance(val, types.StringType), "input must be string")
    # FIXME
    # np.s_ doesn't work for slices starting at end, like
    # >>> a = array([1,2,3,4,5,6])
    # >>> a[-2:]
    # array([5, 6])
    # >>> a[np.s_[-2:]]
    # array([], dtype=int64)
    # >>> np.s_[-2:]
    # slice(9223372036854775805, None, None)
    if val.stip().startswith('-'):
        raise StandardError("Some minus slices (e.g -2:) not supported")
    # This eval() trick works but seems hackish. Better ideas, anyone?
    return eval('np.s_[%s]' %val)


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
        raise StandardError("illegal input value '%s'" %frepr(val))
    return ret


def ffloat(st):
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
    assert_cond(isinstance(st, types.StringType), "`st` must be string")
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
        rex = re.compile(r'\s*([+-]*[0-9\.]+)d([+-]*)([0-9]+)([_]*.*)')
        m = rex.match(st)
        if m is None:
            raise ValueError("no match on string '%s'" %st)
        if m.group(4).strip() != '':
            verbose("[ffloat] WARNING: skipping kind '%s' in string '%s'" 
                %(m.group(4), st))
        ss = "%se%s%s" %m.groups()[:-1]
        return float(ss)


def frepr(var, ffmt="%.16e"):
    """Similar to Python's repr(), but 
    * Return floats formated with `ffmt` if `var` is a float.
    * If `var` is a string, e.g. 'lala', it returns 'lala' not "'lala'" as
      Python's repr() does.
    
    args:
    -----
    var : almost anything (str, None, int, float)
    ffmt : format specifier for float values
    
    examples:
    ---------
    1     -> '1'
    1.0   -> '1.000000000000000e+00' 
    None  -> 'None'
    'abc' -> 'abc' (repr() does: 'abc' -> "'abc'")
    """
    if isinstance(var, types.FloatType):
        return ffmt %var
    elif isinstance(var, types.StringType):
        return var
    else:
        return repr(var)


def seq2str(t):
    """
    (1,2,3) -> "1 2 3"
    """
    return " ".join(map(str, t))


def str2tup(s, func=int):
    """
    "1 2 3" -> (func('1'), func('2'), func('3')) 
    """
    return tuple(map(func, s.split()))


def fix_eps(arr, eps=1.5*np.finfo(float).eps, copy=True):
    """Set values of arr to zero where abs(arr) <= eps.

    args:
    ----
    arr : numpy nd array
    eps : float eps
    copy : bool
        return copy of arr

    returns:
    --------
    numpy nd array
    """
    _arr = arr.copy() if copy else arr
    _arr[np.abs(_arr) <= eps] = 0.0
    return _arr


def str_arr(arr, fmt='%.16e', delim=' '*4, zero_eps=True):
    """Convert array `arr` to nice string representation for printing.
    
    args:
    -----
    arr : array_like, 1d or 2d array
    fmt : string, format specifier, all entries of arr are formatted with that
    delim : string, delimiter
    zero_eps : bool
        Print values as 0.0 where |value| < eps

    returns:
    --------
    string

    examples:
    ---------
    >>> a=rand(3)
    >>> str_arr(a, fmt='%.2f')
    '0.26 0.35 0.97'
    >>> a=rand(2,3)
    >>> str_arr(a, fmt='%.2f')
    '0.13 0.75 0.39\n0.54 0.22 0.66'

    >>> print str_arr(a, fmt='%.2f')
    0.13 0.75 0.39
    0.54 0.22 0.66
    
    notes:
    ------
    Essentially, we replicate the core part of np.savetxt.
    """
    arr = np.asarray(arr)
    _arr = fix_eps(arr) if zero_eps else arr
    if _arr.ndim == 1:
        return delim.join([fmt]*_arr.size) % tuple(_arr)
    elif _arr.ndim == 2:
        _fmt = delim.join([fmt]*_arr.shape[1])
        lst = [_fmt % tuple(row) for row in _arr]
        return '\n'.join(lst)
    else:
        raise ValueError('rank > 2 arrays not supported')


def normalize(a):
    """Normalize array by it's max value. Works also for complex arrays.

    example:
    --------
    >>> a=np.array([3+4j, 5+4j])
    >>> a
    array([ 3.+4.j,  5.+4.j])
    >>> a.max()
    (5.0+4.0j)
    >>> a/a.max()
    array([ 0.75609756+0.19512195j,  1.00000000+0.j ])
    """
    return a / a.max()


def norm_int(y, x, area=1.0):
    """Normalize integral area of y(x) to `area`.
    
    args:
    -----
    x,y : numpy 1d arrays
    area : float

    returns:
    --------
    scaled y

    notes:
    ------
    The argument order y,x might be confusing. x,y would be more natural but we
    stick to the order used in the scipy.integrate routines.
    """
    # First, scale x and y to the same order of magnitude before integration.
    # This may be necessary to avoid numerical trouble if x and y have very
    # different scales.
    fx = 1.0 / np.abs(x).max()
    fy = 1.0 / np.abs(y).max()
    sx = fx*x
    sy = fy*y
##    # Don't scale.
##    fx = fy = 1.0
##    sx, sy = x, y
    # Area under unscaled y(x).
    _area = simps(sy, sx) / (fx*fy)
    return y*area/_area


def deriv_fd(y, x=None, n=1):
    """n-th derivative for 1d arrays of possibly nonuniformly sampled data.
    Returns matching x-axis for plotting. Simple finite differences are used:
    f'(x) = [f(x+h) - f(x)] / h
    
    args:
    -----
    x,y : 1d arrays of same length
        if x=None, then x=arange(len(y)) is used
    n : int
        order of the derivative
    
    returns:
    --------
    xd, yd
    xd : 1d array, (len(x)-n,)
        matching x-axis
    yd : 1d array, (len(x)-n,)
        n-th derivative of y at points xd

    notes:
    ------
    n > 1 (e.g. n=2 -> 2nd derivative) is done by
    recursive application. 
    
    For nonuniformly sampled data, errors blow up quickly. You are strongly
    engouraged to re-sample the data with constant h (e.g. by spline
    interpolation first). Then, derivatives up to ~ 4th order are OK for
    plotting, not for further calculations (unless h is *very* small)!.
    If you need very accurate derivatives, look into NR, 3rd ed., ch. 5.7 and
    maybe scipy.derivative(). 

    Each application returns len(x)-1 points. So for n=3, the returned x and y
    have len(x)-3.

    example:
    --------
    >>> x=sort(rand(100)*10); y=sin(x); plot(x,y, 'o-'); plot(x,cos(x), 'o-')
    >>> x1,y1=deriv_fd(y,x,1) # cos(x)
    >>> x2,y2=deriv_fd(y,x,2) # -sin(x)
    >>> plot(x1, y1, lw=2) # cos(x)
    >>> plot(x2, -y2, lw=2) # sin(x)
    >>> x=linspace(0,10,100); y=sin(x); plot(x,y, 'o-'); plot(x,cos(x), 'o-')
    >>> ...
    
    see also:
    ---------
    numpy.diff()
    """
    assert n > 0, "n <= 0 makes no sense"
    if n > 1:
        x,y = deriv(y, x, n=1)
        return deriv(y, x, n=n-1)
    else:            
        if x is None:
            x = np.arange(len(y))
        dx = np.diff(x)
        return x[:-1]+.5*dx, np.diff(y)/dx


def deriv_spl(y, x=None, xnew=None, n=1, k=3, fullout=True):
    """n-th derivative for 1d arrays of possibly nonuniformly sampled data.
    Returns matching x-axis for plotting. Splines are used.
    
    args:
    -----
    x,y : 1d arrays of same length
        if x=None, then x=arange(len(y)) is used
    xnew : {None, 1d array)
        x-axis to evaluate the derivative, if None then xnew=x
    n : int
        order of the derivative, can only be <= k 
    k : int
        order of the spline; k=n is not recommended, use at least k=n+1
    fullout : bool
        return xd, yd or just yd

    returns:
    --------
    if fullout:
        xd, yd
    else:
        yd
    xd : 1d array, (len(x) or len(xnew),)
    yd : 1d array, (len(x) or len(xnew),)
        n-th derivative of y at points xd
    
    notes:
    ------
    xd is actually == x or xnew (if x is not None). xd can be returned to match
    the function signature of deriv_fd.
    """
    assert n > 0, "n <= 0 makes no sense"
    if x is None:
        x = np.arange(len(y))
    if xnew is None:
        xnew = x
    yd = splev(xnew, splrep(x, y, s=0, k=k), der=n)
    if fullout:
        return xnew, yd
    else:
        return yd

#-----------------------------------------------------------------------------
# array indexing
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
            `sl` is a list or tuple of slice objects, one for each axis. 
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : bool, return a copy instead of a view
    
    returns:
    --------
    A view into `a` or copy of a slice of `a`.

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
    """
    # The long story
    # --------------
    # 
    # 1) Why do we need that:
    # 
    # # no problem
    # a[5:10:2]
    # 
    # # the same, more general
    # sl = slice(5,10,2)
    # a[sl]
    #
    # But we want to:
    #  - Define (type in) a slice object only once.
    #  - Take the slice of different arrays along different axes.
    # Since numpy.take() and a.take() don't handle slice objects, one would
    # have to use direct slicing and pay attention to the shape of the array:
    #       
    #     a[sl], b[:,:,sl,:], etc ...
    # 
    # We want to use an 'axis' keyword instead. np.r_() generates index arrays
    # from slice objects (e.g r_[1:5] == r_[s_[1:5] ==r_[slice(1,5,None)]).
    # Since we need index arrays for numpy.take(), maybe we can use that? Like
    # so:
    #     
    #     a.take(r_[sl], axis=0)
    #     b.take(r_[sl], axis=2)
    # 
    # Here we have what we want: slice object + axis kwarg.
    # But r_[slice(...)] does not work for all slice types. E.g. not for
    #     
    #     r_[s_[::5]] == r_[slice(None, None, 5)] == array([], dtype=int32)
    #     r_[::5]                                 == array([], dtype=int32)
    #     r_[s_[1:]]  == r_[slice(1, None, None)] == array([0])
    #     r_[1:]
    #         ValueError: dimensions too large.
    # 
    # The returned index arrays are wrong (or we even get an exception).
    # The reason is given below. 
    # Bottom line: We need this function.
    #
    # The reason for r_[slice(...)] gererating sometimes wrong index arrays is
    # that s_ translates a fancy index (1:, ::5, 1:10:2, ...) to a slice
    # object. This *always* works. But since take() accepts only index arrays,
    # we use r_[s_[<fancy_index>]], where r_ translates the slice object
    # prodced by s_ to an index array. THAT works only if start and stop of the
    # slice are known. r_ has no way of knowing the dimensions of the array to
    # be sliced and so it can't transform a slice object into a correct index
    # array in case of slice(<number>, None, None) or slice(None, None,
    # <number>).
    #
    # 2) Slice vs. copy
    # 
    # numpy.take(a, array([0,1,2,3])) or a[array([0,1,2,3])] return a copy of
    # `a` b/c that's "fancy indexing". But a[slice(0,4,None)], which is the
    # same as indexing (slicing) a[:4], return *views*. 
    
    if axis is None:
        slices = sl
    else: 
        # Note that these are equivalent:
        #   a[:]
        #   a[s_[:]] 
        #   a[slice(None)] 
        #   a[slice(None, None, None)]
        #   a[slice(0, None, None)]   
        slices = [slice(None)]*a.ndim
        slices[axis] = sl
    # a[...] can take a tuple or list of slice objects
    # a[x:y:z, i:j:k] is the same as
    # a[(slice(x,y,z), slice(i,j,k))] == a[[slice(x,y,z), slice(i,j,k)]]
    if copy:
        return a[slices].copy()
    else:        
        return a[slices]


def sliceput(a, b, sl, axis=None):
    """The equivalent of a[<slice or index>]=b, but accepts slices objects
    instead of array indices or fancy indexing (e.g. a[:,1:]).
    
    args:
    -----
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None 
            `sl` is a list or tuple of slice objects, one for each axis. 
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
    a[tmp] = b
    return a

#-----------------------------------------------------------------------------
# Some handy file operations.
#-----------------------------------------------------------------------------


def get_filename(fh):
    """Try to get the `name` attribute from file-like objects. If it fails
    (fh=cStringIO.StringIO(), fh=StringIO.StringIO(), fh=gzip.open(), ...), 
    then return a dummy name."""
    try:
        name = fh.name
    except AttributeError:
        name = 'object_%s_pwtools_dummy_filename' %str(fh)
    return name        


def file_read(fn):
    """Open file with name `fn`, return open(fn).read()."""
    fd = open(fn, 'r')
    txt = fd.read()
    fd.close()
    return txt


def file_write(fn, txt):
    """Write string `txt` to file with name `fn`. No check is made wether the
    file exists and/or is nonempty. Yah shalleth know whath thy is doingth.  
    shell$ echo $string > $file """
    fd = open(fn, 'w')
    fd.write(txt)
    fd.close()


def file_readlines(fn):
    """Open file with name `fn`, return open(fn).readlines()."""
    fd = open(fn, 'r')
    lst = fd.readlines()
    fd.close()
    return lst


def fullpath(s):
    """Complete path: absolute path + $HOME expansion."""
    return os.path.abspath(os.path.expanduser(s))


def fullpathjoin(*args):
    return fullpath(os.path.join(*args))


def igrep(pat_or_rex, iterable, func='search'):
    """
    Grep thru strings provided by iterable.next() calls. On each match, yield a
    Match object.

    args:
    -----
    pat_or_rex : regex string or compiled re Pattern object, if string then it
        will be compiled
    iterable : sequence of lines (strings to search) or open file object or in 
        general anything that can be iterated over and yields a string (i.e. an
        object that has a next() method)
    func : string, the used re function for matching, e.g. 'match' for re.match
        functionallity, 'search' for re.search

    returns:
    --------
    generator object which yields Match objects

    notes:
    ------
    This function is similar to re.findall()
    
    Difference to grep(1):
        Similar to the shell grep(1) utility, but there is a subtle but
        important difference: grep(1) returns the whole line, not the match
        itself, while the group() method of Match Objects returns ONLY the
        match itself. If you want default grep(1) behavior, use
        "^.*<pattern>.*$" to explicitly match the whole line.         
        
            $ egrep '<pattern>' file.txt
            >>> for m in igrep(r'^.*<pattern>.*$', open('file.txt'), 'search'): print m.group()
            
            $ egrep -o '<pattern>' file.txt
            >>> for m in igrep(r'<pattern>', open('file.txt'), 'search'): print m.group()
   
    Match, Search and grouping:
        In the first example, 'match' would also work since you are matching
        the whole line anyway. 
        
        From the python re module docs:
            match(string[, pos[, endpos]])
                If zero or more characters at the beginning of string match
                this regular expression, return a corresponding MatchObject
                instance. Return None if the string does not match the pattern;
                note that this is different from a zero-length match.

                Note: If you want to locate a match anywhere in string, use
                search() instead. 

        One can directly access match groups, like grep -o:
            
            $ egrep -o '<pattern>' file.txt
            >>> for m in igrep(r'<pattern>', open('file.txt'), 'search'): print m.group()
        
        More explicitly, but no gain here:

            >>> for m in igrep(r'(<pattern>)', open('file.txt'), 'search'): print m.group(1)
                                 ^         ^                                             ^^^ 
        If you want to extract a substring using 'match', you have to match the
        whole line and group the pattern with ()'s to access it.       
            
            >>> for m in igrep(r'^.*(<pattern>).*$', open('file.txt'), 'match'): print m.group(1)

        One possilbe other way would be to call grep(1) & friends thru
            print subprocess.Popen("egrep -o '<pattern>' file.txt", shell=True,
                                    stdout=PIPE).communicate()[0] 
        
        It's generally a good idea to use raw strings: r'<pattern>' instead of
        'pattern'.

    Speed:
        * to get as fast as grep(1), use the idiom
            for m in igrep(r'<pattern>', open('file.txt)', 'match'):
                [do stuff with m]
        * 'match' is (much) faster than 'search', so if you want speed, your
          regexes might have to get more complicated in order for 'match' to work
        * reading in the whole file first, i.e. iterating over lines =
          file_readlines('file.txt') instead of open('file.txt') can speed things
          up if files are small, there is no benefit for big files

    example:
    --------
    # If a line contains at least three numbers, grep the first three.
    >>> !cat file.txt
    a b 11  2   3   xy
    b    4  5   667
    c    7  8   9   4 5
    lol 2
    foo
    >>> !egrep -o '(\s+[0-9]+){3}?' file.txt | tr -s ' '
     11 2 3
     4 5 667
     7 8 9
    >>> fd = open('file.txt')     
    >>> for m in igrep(r'(\s+[0-9]+){3}?', fd, 'search'): print m.group().strip() 
    11  2   3
    4  5   667
    7  8   9
    >>> fd.seek(0)
    >>> for m in igrep(r'((\s+[0-9]+){3}?)', fd, 'search'): print m.group(1).strip() 
    11  2   3
    4  5   667
    7  8   9
    >>> fd.seek(0)
    >>> for m in igrep(r'^.*((\s+[0-9]+){3}?).*$', fd, 'match'): print m.group(1).strip()
    11  2   3
    4  5   667
    7  8   9
    # Put numbers directly into numpy array.
    >>> fd.seek(0)
    >>> ret = igrep(r'(\s+[0-9]+){3}?', fd, 'search') 
    >>> array([m.group().split() for m in ret], dtype=float)
    array([[  11.,    2.,    3.],
           [   4.,    5.,  667.],
           [   7.,    8.,    9.]])
    >>> fd.close()
    """
    if not hasattr(iterable, 'next'):
        raise ValueError("input has no next() method, try iter(...)")

    if isinstance(pat_or_rex, types.StringType):
        rex = re.compile(pat_or_rex)
    else:
        rex = pat_or_rex
    # rex.match(), rex.search(), ... anything that has the signature of
    # {re|Pattern object).match() and returns None or a Match object
    rex_func = getattr(rex, func)        
    for line in iterable:
        match = rex_func(line)
        if match is not None:
            yield match


def mgrep(*args,  **kwargs):
    """Like igrep, but returns a list of Match Objects."""
    return [m for m in igrep(*args, **kwargs)]


def tgrep(*args,  **kwargs):
    """Like igrep, but returns a list of text strings, each is a match."""
    return [m.group() for m in igrep(*args, **kwargs)]


def template_replace(txt, dct, conv=False, warn_mult_found=True,
                     warn_not_found=True, disp=True, mode='dct'):
    """Replace placeholders dct.keys() with string values dct.values() in a
    text string. This function adds some bells and whistles such as warnings
    in case of not-found placeholders and whatnot. 
    
    args:
    -----
    txt : string
    dct : dictionary with placeholders (keys) and values to replace them
    conv : bool, convert `dct` values to strings with str()
    warn_mult_found : bool, warning if a key is found multiple times in `txt`
    warn_not_found : bool, warning if a key is NOT found in `txt`
    disp : tell which keys have been replaced
    mode: str, {'dct', 'txt'}, placeholder mode
        'dct' : Dictionary mode. Placeholders are of special Python dictionary
            string replacement form: '%(<name>)<format_str>', e.g. '%(foo)s'
            and dct.keys() must be normal strings, e.g. 'foo'.
            dct.values() can be anything. The conversion to a string is done at
            replacement time and determined by the <format_str>. This
            effectively does `txt % dct`. This method is faster, uses Python
            standard syntax and is therefore default.  
        'txt' : Text mode. Placeholders in `txt` and keys in `dct` are the
            exact same arbitrary string (e.g. 'XXXFOO' in both). Here,
            dct.values() must be strings. If not, use conv=True to
            automatically convert them to strings, but note that this is
            limited since only str(<val>) is used.
    
    returns:
    --------
    new string
    
    example:
    --------
    >>> txt = file_read('file.txt') 
    >>> dct = {'XXXONE': 1, 'XXXPI': '%.16e' %math.pi}
    >>> new_txt = template_replace(txt, dct, conv=True, mode='txt')
    >>>
    >>> txt = 'XXXONE  XXXPI'                            
    >>> dct = {'XXXONE': '1', 'XXXPI': '%.16e' %math.pi}
    >>> template_replace(txt, dct, mode='txt')
    >>> '1  3.1415926535897931e+00'
    >>>
    >>> txt = '%(one)s  %(pi).16e'; dct = {'one': 1, 'pi': math.pi}
    >>> template_replace(txt, dct)
    >>> '1  3.1415926535897931e+00'
    >>> txt % dct
    >>> '1  3.1415926535897931e+00'
    """
    if isinstance(txt, types.DictType):
        raise ValueError("1st arg is a dict. You probably use the old syntax. "
                         "The new syntax in func(txt, dct) instead of "
                         "func(dct, txt)")
    is_txt_mode = False
    is_dct_mode = False
    if mode == 'dct':
        is_dct_mode = True
        if conv:
            print("template_replace: Warning: `conv=True` is ignored if "
                  "mode=='dct', instead use proper format strings in your "
                  "placeholders")
    elif mode == 'txt':
        is_txt_mode = True
    else:
        raise StandardError("mode must be 'txt' or 'dct'")
    
    # This is a copy. Each txt.replace() returns an additional copy. We need
    # that if we loop over dct.iteritems() and sucessively replace averything.
    if is_txt_mode: 
        new_txt = txt
    for key, val in dct.iteritems():
        if is_dct_mode:
            # The key is '%(foo)s', but searching for '%(foo)' must suffice,
            # since we don't know the format string, in this case 's', in
            # `txt`.
            tst_key = '%'+ '(%s)' %key
        else:
            tst_key = key
        if tst_key in txt:
            if is_txt_mode:
                if conv:
                    val = str(val)
                else:                    
                    if not isinstance(val, types.StringType):
                        raise StandardError("dict vals must be strings: "
                                        "key: '%s', val: " %key + str(type(val)))
            if warn_mult_found:                    
                cnt = txt.count(tst_key)
                if cnt > 1:
                    print("template_replace: warning: key '%s' found %i times"
                    %(tst_key, cnt))
            if is_txt_mode:                    
                new_txt = new_txt.replace(key, val)
            if disp:
                print("template_replace: %s -> %s" %(key, val))
        else:
            if warn_not_found:
                print "template_replace: key not found: %s" %tst_key
    if is_dct_mode:
        new_txt = txt % dct
    return new_txt


def file_template_replace(fn, dct, bak='', **kwargs):
    """Replace placeholders in file `fn`.

    args:
    -----
    fn : str
        Filename
    dct : dict
        Replacement rules
    bak : str
        '' : no backup is done
        '<str>' : `fn` is backed up to "fn<str>"
    kwargs : kwargs to template_replace()

    example:
    --------
    dct = {'xxx': 'foo', 'yyy': 'bar'}
    fn = 'bla.txt'
    file_template_replace(fn, dct, '.bak', mode='txt')
    
    This the same as:
    shell$ sed -i.bak -r -e 's/xxx/foo/g -e 's/yyy/bar/g' bla.txt"""
    txt = template_replace(file_read(fn), dct, **kwargs)
    if bak != '':
        shutil.copy(fn, fn + bak)                
    file_write(fn, txt)


def backup(src, prefix='.'):
    """Backup (copy) `src` to <src><prefix><num>, where <num> is an integer
    starting at 0 which is incremented until there is no destination with that
    name.
    
    Symlinks are handled by shutil.copy() for files and shutil.copytree() for
    dirs. In both cases, the content of the file/dir pointed to by the link is
    copied.

    args:
    -----
    src : str
        name of file/dir to be copied
    prefix : str, optional
    """
    if os.path.isfile(src):
        copy = shutil.copy 
    elif os.path.isdir(src):
        copy = shutil.copytree
    else:
        raise StandardError("source '%s' is not file or dir" %src)
    idx = 0
    dst = src + '%s%s' %(prefix,idx)
    while os.path.exists(dst):
        idx += 1
        dst = src + '%s%s' %(prefix,idx)
    # sanity check
    if os.path.exists(dst):
        raise StandardError("destination '%s' exists" %dst)
    else:        
        copy(src, dst)

#-----------------------------------------------------------------------------
# Dictionary tricks
#-----------------------------------------------------------------------------

def dict2str(dct):
    """Nicer than simply __repr__."""
    st = ""    
    for key, val in dct.iteritems():
        st += "%s: %s\n" %(key, repr(val))
    return st        

# backward compat only
def print_dct(dct):
    print dict2str(dct)

def dict2class(dct, name='Dummy'):
    """
    Convert a dict to a class.

    example:
    --------
    >>> dct={'a':1, 'b':2}
    >>> dct2class(dct, 'Foo')
    <Foo instance at 0x3615ab8>
    >>> dct2class(dct, 'Bar')
    <Bar instance at 0x3615b48>
    >>> dct2class(dct, 'Bar').__dict__
    {'a':1, 'b':2}
    """
    class Dummy:
        pass
    cl = Dummy()
    cl.__dict__.update(dct)
    cl.__class__.__name__ = name
    return cl


#-----------------------------------------------------------------------------
# Sequence tricks
#-----------------------------------------------------------------------------

def is_seq(seq):
    """Test if `seq` is some kind of sequence."""
    # Exclude cases which are iterable but that we still don't like. In fact,
    # we wish to catch list, tuple, numpy array.
    if isinstance(seq, types.StringType) or \
       isinstance(seq, types.FileType):
       return False
    else:        
        try:
            x=iter(seq)
            return True
        except:
            return False


def iflatten(seq):
    """Flatten a sequence. After
    matplotlib.cbook.flatten(). Returns an generator object."""
    for item in seq:
        if not is_seq(item):
            yield item
        else:
            for subitem in flatten(item):
                yield subitem


def flatten(seq):
    """Same as iflatten(), but returns a list."""
    return [x for x in iflatten(seq)]

#-----------------------------------------------------------------------------
# Child processes & shell calls
#-----------------------------------------------------------------------------

def system(call, wait=True):
    """Fire up shell commamd line `call`. 
    
    args:
    -----
    call: str (example: 'ls -l')
    wait : bool
        False: Don't wait for `call` to terminate.
            This can be used to spawn multiple processes concurrently. This is
            identical to calling os.system(call) (as opposed to ret=os.system(call).
        True: This is identical to calling ret=os.system(call).
    """
    p = subprocess.Popen(call, shell=True)
    if wait:
        os.waitpid(p.pid, 0)

def permit_sigpipe():
    """Helper for subprocess.Popen(). Handle SIGPIPE. To be used as preexec_fn.

    notes:
    ------
    Some cases like:
        >>> cmd = r"grep pattern /path/to/very_big_file | head -n1"
        >>> pp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
        ...                       stderr=subprocess.PIPE)
        >>> out,err = pp.communicate()
    
    sometimes end with a broken pipe error:
        "grep: writing output: Broken pipe"
    They run fine at the bash prompt, while failing with Popen. The reason is
    that they actually "kind of" fail in the shell too, namely, SIGPIPE [1,2].
    This can be seen by runing the call in strace "$ strace grep ...". Popen
    chokes on that. The solution is to ignore SIGPIPE.

    refs:
    -----
    [1] http://mail.python.org/pipermail/tutor/2007-October/058042.html
    [2] http://article.gmane.org/gmane.comp.python.devel/88798/
    """
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def backtick(call):
    """Convenient shell backtick replacement with gentle error handling.

    example:
    --------
    >>> print backtick('ls -l')
    """
    pp = subprocess.Popen(call, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=permit_sigpipe)
    out,err = pp.communicate()
    if err.strip() != '':
        raise StandardError("Error calling command: '%s'\nError message "
            "follows:\n%s" %(call, err))
    return out            

#-----------------------------------------------------------------------------
# aliases
#-----------------------------------------------------------------------------

fpj = fullpathjoin
pj = os.path.join
# backw. compat
grep = mgrep 
tup2str = seq2str
