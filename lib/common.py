# common.py
#
# File operations, common system utils and other handy tools that could as well
# live in the std lib.
#
# Steve Schmerler 2009 <mefx@gmx.net>
#

import types
import os
import subprocess
import re
import shutil
import re
import hashlib
import ConfigParser

from decorators import add_func_doc


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
# Type converters
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


def tup2str(t):
    """
    (1,2,3) -> "1 2 3"
    """
    return " ".join(map(str, t))


def str2tup(s, func=int):
    """
    "1 2 3" -> (func('1'), func('2'), func('3')) 
    """
    return tuple(map(func, s.split()))



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


def fileo(val, mode='r', force=False):
    """Return open file object with mode `mode`. Handles also gzip'ed files.
    Non-empty files are protected. File objects are just passed through, not
    modified.

    args:
    -----
    val : str or file object
    mode : file mode (everything that Python's open() can handle)
    force : bool, force to overwrite non-empty files
    """
    if isinstance(val, types.StringType):
        if os.path.exists(val): 
            if not os.path.isfile(val):
                raise ValueError("argument '%s' exists but is no file" %val)
            if ('w' in mode) and (not force) and (os.path.getsize(val) > 0):
                raise StandardError("file '%s' not empty, won't ovewrite, use "
                    "force=True" %val)
        if val.endswith('.gz'):
            import gzip
            ret =  gzip.open(val, mode)
            # Files opened with gzip don't have a 'name' attr.
            if not hasattr(ret, 'name'):
                ret.name = fullpath(val)
            return ret                
        else:
            return open(val, mode)
    elif isinstance(val, types.FileType):
        if val.closed:
            raise StandardError("need an open file") 
        elif val.mode != mode:
            raise ValueError("mode of file '%s' is '%s', must be '%s'" 
                %(val.name, val.mode, mode))
        return val


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
    This function could also live in Python's itertools module.
    
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
        But that's not really pythonic, eh? :)
        
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
    >>> for m in igrep(r'(\s+[0-9]+){3}?', open('file.txt'), 'search'): print m.group().strip() 
    11  2   3
    4  5   667
    7  8   9
    >>> for m in igrep(r'((\s+[0-9]+){3}?)', open('file.txt'), 'search'): print m.group(1).strip() 
    11  2   3
    4  5   667
    7  8   9
    >>> for m in igrep(r'^.*((\s+[0-9]+){3}?).*$', open('file.txt'), 'match'): print m.group(1).strip()
    11  2   3
    4  5   667
    7  8   9
    # Put numbers directly into numpy array.
    >>> ret=igrep(r'(\s+[0-9]+){3}?', fd, 'search') 
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
            effecticly does `txt % dct`. This method is faster, uses Python
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


class FileTemplate(object):
    """Class to represent a template file in parameter studies.
    
    example
    -------

    This will take a template file calc.templ/pw.in, replace the placeholders
    "@prefix@" and "@ecutwfc@" with some values and write the file to
    calc/0/pw.in .

    >>> templ = FileTemplate(basename='pw.in',
    >>>                      keys=['prefix', 'ecutwfc'],
    >>>                      dir='calc.templ',
    >>>                      func=lambda x: "@%s@" %x)
    >>>
    >>> dct = {}
    >>> dct['prefix'] = 'foo_run_1'
    >>> dct['ecutwfc'] = 23.0
    >>> templ.write(dct, 'calc/0')
    >>>
    # or with SQL foo in a parameter study
    >>>
    >>> from sql import SQLEntry
    >>> dct = {}                     
    >>> dct['prefix']  = SQLEntry(sql_type='text',  sql_val='foo_run_1')
    >>> sct['ecutwfc'] = SQLEntry(sql_type='float', sql_val=23.0)
    >>> templ.writesql(dct, 'calc/0')
    
    placeholders
    ------------
    The default placeholder is "XXX<KEY>", where <KEY> is the upercase version
    of an entry in `keys`. See _default_get_placeholder().
    """
    
    # "keys" is needed if several templates exist whose write*() methods are
    # called with the same "dct" or "sql_record". This is common in parameter
    # studies, where we have ONE sql_record holding all parameters which are
    # vaired in the current run, but the parameters are spread across several
    # template files.
    #
    # If needed, implement that "keys" in __init__ is optional. Then, if
    # keys=None, write*() should simply write all entries in "dct" or
    # "sql_record". 
    
    def __init__(self, keys, basename, dir='calc.templ',
                 func=None, phfunc=None):
        """
        args
        ----
        keys : list of strings
            Each string is a key. Each key is connected to a placeholder in the
            template. See func.
        basename : string
            The name of the template file and target file.
            example: basename = pw.in
                template = calc.templ/pw.in
                target   = calc/0/pw.in
        dir : dir where the template lives (e.g. calc.templ)
        func : callable
            A function which takes a string (key) and returns a string, which
            is the placeholder corresponding to that key.
            example: (this is actually default)
                key = "lala"
                placeholder = "XXXLALA"
                func = lambda x: "XXX" + x.upper()
        """
        # For backward compat. Remove later.
        if phfunc is not None:
            print("Grep: Warning: 'phfunc' kwarg renamed to 'func'. Use that "
                  "in the future.")
            if func is None:
                func = phfunc
            else:
                raise StandardError("got 'func' and 'phfunc' kwargs, cannot "
                                    "use both")

        self.keys = keys
        self.dir = dir
        
        # We hardcode the convention that template and target files live in
        # different dirs and have the same name ("basename") there.
        #   template = <dir>/<basename>
        #   target   = <calc_dir>/<basename>
        # e.g.
        #   template = calc.templ/pw.in
        #   target   = calc/0/pw.in
        # Something like
        #   template = ./pw.in.templ
        #   target   = ./pw.in
        # is not possible.
        self.basename = basename
        self.filename = pj(self.dir, self.basename)
        
        self._get_placeholder = self._default_get_placeholder if func is None \
                                else func
        
        # copy_only : bypass reading the file and passing the text thru the
        # replacement machinery and getting the text back, unchanged. While
        # this works, it is slower and useless.
        if keys != []:
            self.txt = self._get_txt()
            self._copy_only = False
        else:
            self.txt = None
            self._copy_only = True
    
    def _default_get_placeholder(self, key):
        return 'XXX' + key.upper()

    def _get_txt(self):
        return file_read(self.filename)
    
    def write(self, dct, calc_dir='calc', type='dct'):
        assert type in ['dct', 'sql'], "Wrong 'type' kwarg, use 'dct' \
                                       or 'sql'"
        tgt = pj(calc_dir, self.basename)
        if self._copy_only:    
            verbose("write: ignoring input, just copying file: %s -> %s"
                    %(self.filename, tgt))
            shutil.copy(self.filename, tgt)
        else:            
            rules = {}
            for key in self.keys:
                if type == 'dct':
                    rules[self._get_placeholder(key)] = dct[key]
                elif type == 'sql':                    
                    # dct = sql_record, a list of SQLEntry's
                    rules[self._get_placeholder(key)] = dct[key].file_val
                else:
                    raise StandardError("'type' must be wrong")
            file_write(tgt, template_replace(self.txt, rules, mode='txt',
                                             conv=True))
    
    def writesql(self, sql_record, calc_dir='calc'):
        self.write(sql_record, calc_dir=calc_dir, type='sql')



#-----------------------------------------------------------------------------
# Dictionary tricks
#-----------------------------------------------------------------------------

def print_dct(dct):
    for key, val in dct.iteritems():
        print "%s: %s" %(key, str(val))


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


def backtick(call):
    """Convenient shell backtick replacement with gentle error handling.

    example:
    --------
    >>> print backtick('ls -l')
    """
    out,err = subprocess.Popen(call, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
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
