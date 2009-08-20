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
        
            $ egrep -o '<pattern>' file.txt
            >>> for m in igrep(r'<pattern>', open('file.txt'), 'search'): print m.group()
        
            $ egrep '<pattern>' file.txt
            >>> for m in igrep(r'^.*<pattern>.*$', open('file.txt'), 'search'): print m.group()
   
    Match, Search and grouping:
        In the latter example, 'match' would also work since you are matching
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
        
        It's generally a good idoa to use raw strings: r'<pattern>' instead of
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

def raw_template_replace(txt, dct, conv=False, warn_mult_found=True,
                         warn_not_found=True, disp=True):
    """Replace placeholders dct.keys() with string values dct.values() in a
    text string. 
    
    args:
    -----
    txt : string
    dct : dictionary 
    conv : bool, convert `dct` values to strings with str()
    warn_mult_found : bool, warning if a key is found multiple times in `txt`
    warn_not_found : bool, warning if a key is NOT found in `txt`
    disp : tell which keys hav been replaced
    
    returns:
    --------
    new string

    notes:
    ------
    `txt` us usually a read text file (txt=fd.read()).  Although we use
    txt.replace(), this method ~ 4 times faster then looping
    over lines in fd. But: only as long as `txt` fits entirely into memory.
    """
    if isinstance(txt, types.DictType):
        raise ValueError("1st arg is a dict. You probably use the old syntax. "
                         "The new syntax in func(txt, dct) instead of "
                         "func(dct, txt)")
    # This is a pointer. Each txt.replace() returns a copy.
    new_txt = txt
    for key, val in dct.iteritems():
        if key in new_txt:
            if val is None:
                print "template_replace: value for key '%s' is None, skipping" %key
                continue
            if conv:
                val = str(val)
            if not isinstance(val, types.StringType):
                raise StandardError("dict vals must be strings: "
                                    "key: '%s', val: " %key + str(type(val)))
            if warn_mult_found:                    
                cnt = txt.count(key)
                if cnt > 1:
                    print("template_replace: warning: key '%s' found %i times"
                    %(key, cnt))
            new_txt = new_txt.replace(key, val)
            if disp:
                print("template_replace: %s -> %s" %(key, val))
        else:
            if warn_not_found:
                print "template_replace: key not found: %s" %key
    return new_txt

#-----------------------------------------------------------------------------

def template_replace(txt, dct, warn=True):
    """Replace placeholders in `txt`. Print only successful replaces and
    warnings (unless warn = False, then don't print warnings)."""
    if isinstance(txt, types.DictType):
        raise ValueError("1st arg is a dict. You probably use the old syntax. "
                         "The new syntax in func(txt, dct) instead of "
                         "func(dct, txt)")
    return raw_template_replace(txt, dct, conv=True, warn_mult_found=warn,
        warn_not_found=warn, disp=True)

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
    file_template_replace(fn, dct, '.bak')
    
    This the same as:
    shell$ sed -i.bak -r -e 's/xxx/foo/g -e 's/yyy/bar/g' bla.txt"""
    txt = template_replace(file_read(fn), dct, **kwargs)
    if bak != '':
        shutil.copy(fn, fn + bak)                
    file_write(fn, txt)

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
