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
import cPickle
import copy
import numpy as np

from pwtools.verbose import verbose
# backward compat, remove later
from pwtools.num import normalize, vlinspace, norm_int, deriv_fd, deriv_spl, \
    _splroot, findmin, findroot, slicetake, sliceput, Spline


def assert_cond(cond, string=None):
    """Use this instead of `assert cond, string`. It's been said on
    numpy-discussions that the assert statement shouldn't be used to test user
    input in functions b/c with `python ... -O0` or __debug__ not beeing
    defined, the statement is not tested.
    
    Parameters
    ----------
    cond : bool
        True : None is returned
        False : exception is raised
    string : str
    
    Examples
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
    
    Parameters
    ----------
    config : ConfigParser object
    info : dict of dicts, see io.writearr()

    Returns
    -------
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
    
    Parameters
    ----------
    val : string

    Examples
    --------
    '3'     -> 3
    '3:'    -> slice(3, None, None)
    '-2:'   -> slice(-2, None, None)
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
    # XXX This is fixed in numpy 1.5.1:
    #   https://github.com/numpy/numpy/commit/9089036b
    # np.s_ doesn't work for slices starting at end, like
    # >>> a = array([1,2,3,4,5,6])
    # >>> a[-2:]
    # array([5, 6])
    # >>> a[np.s_[-2:]]
    # array([], dtype=int64)
    # >>> np.s_[-2:]
    # slice(9223372036854775805, None, None)
    if val.strip().startswith('-'):
        if np.s_[-2:] != slice(-2, None, None): 
            raise StandardError("Some minus slices (e.g -2:) not supported "
                "by your numpy (probably old version). Use "
                "[<start>[:<step>]:<end>] as workaround.")
    # This eval() trick works but seems hackish. Better ideas, anyone?
    return eval('np.s_[%s]' %val)


def tobool(val):
    """Convert `val` to boolean value True or False.
        
    Parameters
    ----------
    val : bool, string, integer
        '.true.', '1', 'true',  'on',  'yes', integers != 0 -> True
        '.false.','0', 'false', 'off', 'no',  integers == 0 -> False
    
    Returns
    -------
    True or False

    Notes
    -----
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

    Parameters
    ----------
    st : string

    Returns
    -------
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
    """Similar to Python's repr(), but return floats formated with `ffmt` if
    `var` is a float.

    If `var` is a string, e.g. 'lala', it returns 'lala' not "'lala'" as
    Python's repr() does.
    
    Parameters
    ----------
    var : almost anything (str, None, int, float)
    ffmt : format specifier for float values
    
    Examples
    --------
    >>> frepr(1)
    '1'
    >>> frepr(1.0) 
    '1.000000000000000e+00' 
    >>> frepr(None)
    'None'
    >>> # Python's repr() does: 'abc' -> "'abc'"
    >>> frepr('abc')
    'abc' 
    """
    if isinstance(var, types.FloatType):
        return ffmt %var
    elif isinstance(var, types.StringType):
        return var
    else:
        return repr(var)


def seq2str(seq, func=str, sep=' '):
    """(1,2,3) -> "1 2 3" """
    return sep.join(map(func, seq))


def str2seq(st, func=int, sep=None):
    """ "1 2 3" -> [func('1'), func('2'), func('3')]"""
    if sep is None:
        return map(func, st.split())
    else:
        return map(func, st.split(sep))


def str2tup(*args, **kwargs):
    return tuple(str2seq(*args, **kwargs))


def fix_eps(arr, eps=1.5*np.finfo(float).eps, copy=True):
    """Set values of arr to zero where abs(arr) <= eps.

    Parameters
    ----------
    arr : numpy nd array
    eps : float eps
    copy : bool
        return copy of arr

    Returns
    -------
    numpy nd array
    """
    _arr = arr.copy() if copy else arr
    _arr[np.abs(_arr) <= eps] = 0.0
    return _arr


def str_arr(arr, fmt='%.16e', delim=' '*4, zero_eps=True):
    """Convert array `arr` to nice string representation for printing.
    
    Parameters
    ----------
    arr : array_like, 1d or 2d array
    fmt : string, format specifier, all entries of arr are formatted with that
    delim : string, delimiter
    zero_eps : bool
        Print values as 0.0 where abs(value) < eps

    Returns
    -------
    str

    Notes
    -----
    Essentially, we replicate the core part of np.savetxt.

    Examples
    --------
    >>> a=rand(3)
    >>> str_arr(a, fmt='%.2f')
    '0.26 0.35 0.97'
    >>> a=rand(2,3)
    >>> str_arr(a, fmt='%.2f')
    '0.13 0.75 0.39\\n0.54 0.22 0.66'
    >>> print str_arr(a, fmt='%.2f')
    0.13 0.75 0.39
    0.54 0.22 0.66
    """
    arr = np.asarray(arr)
    _arr = fix_eps(arr) if zero_eps else arr
    if _arr.ndim == 1:
        return delim.join([fmt]*_arr.size) % tuple(_arr)
    elif _arr.ndim == 2:
        # slightly faster:
        #   nrows = _arr.shape[0]
        #   ncols = _arr.shape[1]
        #   return (delim.join([fmt]*ncols) + '\n')*nrows % tuple(_arr.flatten())
        _fmt = delim.join([fmt]*_arr.shape[1])
        lst = [_fmt % tuple(row) for row in _arr]
        return '\n'.join(lst)
    else:
        raise ValueError('rank > 2 arrays not supported')


#-----------------------------------------------------------------------------
# Some handy file operations.
#-----------------------------------------------------------------------------

def makedirs(path):
    """Same as os.makedirs() but silently skips existing dirs."""
    if not os.path.exists(path):
        os.makedirs(path)

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



def template_replace(txt, dct, conv=False, warn_mult_found=True,
                     warn_not_found=True, disp=True, mode='dct'):
    """Replace placeholders dct.keys() with string values dct.values() in a
    text string. This function adds some bells and whistles such as warnings
    in case of not-found placeholders and whatnot. 
    
    Parameters
    ----------
    txt : string with placeholders
    dct : dictionary with placeholders (keys) and values to replace them
    conv : bool, convert values dct.values() to strings with frepr()
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
            limited since only frepr(<val>) is used.
    
    Returns
    -------
    new string
    
    Examples
    --------
    >>> txt = 'XXXONE  XXXPI'                            
    >>> dct = {'XXXONE': 1, 'XXXPI': math.pi}
    >>> template_replace(txt, dct, conv=True, mode='txt')
    '1  3.1415926535897931e+00'
    >>>
    >>> dct = {'XXXONE': '1', 'XXXPI': '%.16e' %math.pi}
    >>> template_replace(txt, dct, mode='txt')
    '1  3.1415926535897931e+00'
    >>>
    >>> txt = '%(one)s  %(pi).16e'; dct = {'one': 1, 'pi': math.pi}
    >>> template_replace(txt, dct)
    '1  3.1415926535897931e+00'
    >>> 
    >>> txt % dct
    '1  3.1415926535897931e+00'
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
                    val = frepr(val, ffmt="%.16e")
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

    Parameters
    ----------
    fn : str
        Filename
    dct : dict
        Replacement rules
    bak : str
        '' : no backup is done
        '<str>' : `fn` is backed up to "fn<str>"
    kwargs : kwargs to template_replace()

    Examples
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

    Parameters
    ----------
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

    Examples
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
    """Test if `seq` is some kind of sequence, based on calling iter(seq), i.e.
    if the object is iterable.

    Exclude cases which are iterable but that we still don't like: 
        StringTypes = StringType + UnicodeType
        FileType
    UnicodeType is for lists of unicode strings [u'aaa', u'bbb']. In fact, we
    wish to catch list, tuple, numpy array.
    
    Parameters
    ----------
    seq : (nested) sequence of arbitrary objects
    """ 
    if isinstance(seq, types.StringTypes) or \
       isinstance(seq, types.FileType):
       return False
    else:        
        try:
            x=iter(seq)
            return True
        except:
            return False


def iflatten(seq):
    """Flatten a sequence. After matplotlib.cbook.flatten(). Returns an
    generator object."""
    for item in seq:
        if not is_seq(item):
            yield item
        else:
            for subitem in flatten(item):
                yield subitem


def flatten(seq):
    """Same as iflatten(), but returns a list."""
    return [x for x in iflatten(seq)]


def pop_from_list(lst, items):
    """Pop all `items` from `lst` and return a shorter copy of
    `lst`.
    
    Parameters
    ----------
    lst: list
    items : sequence

    Returns
    -------
    lst2 : list
        Copy of `lst` with `items` removed.    
    """
    lst2 = copy.deepcopy(lst)
    for item in items:
        lst2.pop(lst2.index(item))
    return lst2    


def asseq(arg):
    """Assert `arg` to be a sequence. If it already is one (see ``is_seq``)
    then return it, else return a length 1 list."""
    if is_seq(arg):
        return arg
    else:
        return [arg]

#-----------------------------------------------------------------------------
# Child processes & shell calls
#-----------------------------------------------------------------------------

def system(call, wait=True):
    """Fire up shell commamd line `call`. 
    
    Parameters
    ----------
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

    Notes
    -----
    Things like::

        >>> cmd = r"grep pattern /path/to/very_big_file | head -n1"
        >>> pp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
        ...                       stderr=subprocess.PIPE)
        >>> out,err = pp.communicate()
    
    sometimes end with a broken pipe error: "grep: writing output: Broken
    pipe". They run fine at the bash prompt, while failing with Popen. The
    reason is that they actually "kind of" fail in the shell too, namely,
    SIGPIPE [1,2]. This can be seen by runing the call in strace "$ strace grep
    ...". Popen chokes on that. The solution is to ignore SIGPIPE.

    References
    ----------
    .. [1] http://mail.python.org/pipermail/tutor/2007-October/058042.html
    .. [2] http://article.gmane.org/gmane.comp.python.devel/88798/
    """
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def backtick(call):
    """Convenient shell backtick replacement with gentle error handling.

    Examples
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
# pickle
#-----------------------------------------------------------------------------

def cpickle_load(filename):
    return cPickle.load(open(filename, 'rb'))

#-----------------------------------------------------------------------------
# aliases
#-----------------------------------------------------------------------------

fpj = fullpathjoin
pj = os.path.join
tup2str = seq2str
