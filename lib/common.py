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

from decorators import add_func_doc

#-----------------------------------------------------------------------------

def dict2class(dct, name='Dummy'):
    """
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
# Some handy file operations.
#-----------------------------------------------------------------------------

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
        elif not val.mode == mode:
            raise ValueError("mode of file '%s' is '%s', must be '%s'" 
                %(val.name, val.mode, mode))
        return val

#-----------------------------------------------------------------------------

def file_read(fn):
    """Open file with name `fn`, return open(fn).read()."""
    fd = open(fn, 'r')
    txt = fd.read()
    fd.close()
    return txt

#-----------------------------------------------------------------------------

def file_write(fn, txt):
    """Write string `txt` to file with name `fn`. No check is made wether the
    file exists and/or is nonempty. Yah shalleth know whath thy is doingth.  
    shell$ echo $string > $file """
    fd = open(fn, 'w')
    fd.write(txt)
    fd.close()

#-----------------------------------------------------------------------------

def file_readlines(fn):
    """Open file with name `fn`, return open(fn).readlines()."""
    fd = open(fn, 'r')
    lst = fd.readlines()
    fd.close()
    return lst

#-----------------------------------------------------------------------------

def fullpath(s):
    """Complete path: absolute path + $HOME expansion."""
    return os.path.abspath(os.path.expanduser(s))

#-----------------------------------------------------------------------------

def fullpathjoin(*args):
    return fullpath(os.path.join(*args))

#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------

def mgrep(*args,  **kwargs):
    """Like igrep, but returns a list of Match Objects."""
    return [m for m in igrep(*args, **kwargs)]

#-----------------------------------------------------------------------------

def tgrep(*args,  **kwargs):
    """Like igrep, but returns a list of text strings, each is a match."""
    return [m.group() for m in igrep(*args, **kwargs)]

#-----------------------------------------------------------------------------

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
    disp : tell which keys hav been replaced
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
                  "mode=='dct'")
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

#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------

# template hash function
def hash(txt, mod_funcs=[], skip_funcs=[]):
    """
    Calculate a line-order-independent md5 hash of a text file. Do it in a
    special way so that only the hash of the "information contained in the
    file" is computed, not of the whole file. For instance, in most input
    files, whitespaces or the order in which lines appear is not important. 
    
    To get a hash that is independent of the order of the lines, we loop over
    each line, compute it's hash, sum up all hashes and at the end return the
    hash of this sum. 
    
    Additionally, we allow user-defined functions to modify lines (mod_funcs)
    and functions that analyze a line and tell if it should be skipped over. In
    that way, only (possibly modified) lines contribute to the hash. 
    
    In short, we do
        * loop over the lines of the file
        * for each function in `mod_funcs`, apply line=func(line), e.g. remove
          whitespace etc.
        * for each function in `skip_funcs`, apply func(line), if it returns
          True, then that line is skipped (useful to skip over comment lines or
          empty lines)
    args:
    -----
    txt : a string (most likely open(<file>).read())
    mod_funcs : list of functions that take one arg: a string (the line) and
        return a string
    skip_funcs : list of functions that take one arg: a string (the line) and
        return True or False
    
    mod_funcs and skip_funcs shall NOT assume to get lines that are alraedy
    modified in are way. They shall expect lines as read from the file.

    returns:
    --------
    hex string, the hash

    notes:
    ------
    Integer representations of the hex strings of each line, e.g.  
        int(hashlib.md5(...), 16)
    yield large integers. If the result is > sys.maxint, then a long int (as
    opposed to a plain int) is automatically returned. But we use long() right
    away and work only with long. Luckily, these have unlimited precision, so
    there won't be any overflows if we add them up for large files (many
    lines). [1]

    We have no idea if the whole procedure is even mathematically correct etc.
    Use with care.
    
    This function is slow. Don't use on MB-sized files. Especially the usage
    idiom
        hash(file_read(<filename>)) 
    is far from efficient. It reads the whole file into memory and then calls
    txt.splitlines(), which consumes additional memory. This is only important
    for very big files (~5 min for a 650 MB test file). We focus only on
    convenience and correctness right now.

    refs:
    -----
    [1] http://docs.python.org/library/stdtypes.html#numeric-types-int-float-long-complex
    """
    sm = 0
    for line in txt.splitlines():
        skip = False
        for func in skip_funcs:
            if func(line):
                skip=True
                break
        if not skip:
            for func in mod_funcs:
                line = func(line)
            # long(..., 16) converts the kex string into a (long) int
            sm += long(hashlib.md5(line).hexdigest(), 16)
    return hashlib.md5(str(sm)).hexdigest()        

def _add_hash_func_doc(func):
    return add_func_doc(func, doc_func=hash)


# Helper functions which define the "grammar" of the file, i.e. what can be
# removed and skipped over.

# line modifiers

def _rem_ws(line, rex=re.compile(r'\s')):
    r"""Remove whitespace [ \t\n\r\f\v] ."""
    return rex.sub('', line)

def _pwin_rem_comma(line, rex=re.compile(r'(.*),$')):
    """Remove comma at the end of the line."""
    return rex.sub(r'\1', line)

def _pwin_to_lower(line):
    """Even though some identifiers in pw.x input files MUST be upper-case
    (e.g. ATOMIC_POSITIONS), we convert everything to lower for the hash, since
    must identifiers are (as Fortran itself is) case-insensitive."""
    return line.lower()

# skippers, should assume to get lines as read from the file, i.e. NOT already
# modified (e.g. whitespace removed or anything)

def _pwin_match_comment(line, cmt=['!']):
    """Return True if the line starts with one of the strings in
    `cmt`. Here, some compilers allow Fortran 90 comments in the input and
    ignore them. We also do that."""
    if line.strip()[0] in cmt:
        return True
    else:
        return False

def _match_empty(line):
    """Return True if the line contains only whitespace."""
    if line.strip() == '':
        return True
    else:
        return False

@_add_hash_func_doc
def pwin_hash(txt, mod_funcs=[_rem_ws, _pwin_rem_comma, _pwin_to_lower], 
                   skip_funcs=[_pwin_match_comment, _match_empty]):
    """Hash for pw.x input files."""                   
    return hash(txt, mod_funcs=mod_funcs, skip_funcs=skip_funcs)                   

@_add_hash_func_doc
def generic_hash(txt, mod_funcs=[_rem_ws], 
                   skip_funcs=[_match_empty]):
    """Generic hash for text files. Irgores: line order, all whitespaces and
    newlines."""                   
    return hash(txt, mod_funcs=mod_funcs, skip_funcs=skip_funcs)

#-----------------------------------------------------------------------------
# Dictionary tricks
#-----------------------------------------------------------------------------

def print_dct(dct):
    for key, val in dct.iteritems():
        print "%s: %s" %(key, str(val))

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

#-----------------------------------------------------------------------------

def iflatten(seq):
    """Flatten a sequence. After
    matplotlib.cbook.flatten(). Returns an generator object."""
    for item in seq:
        if not is_seq(item):
            yield item
        else:
            for subitem in flatten(item):
                yield subitem

#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------
# aliases
#-----------------------------------------------------------------------------

fpj = fullpathjoin
# backw. compat
grep = mgrep 
