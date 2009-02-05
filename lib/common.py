import types
import os

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
                raise ValueError("argument '%s' exists but is no file")
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
    else:
        raise ValueError("`val` must be string or file object")

#-----------------------------------------------------------------------------

def file_read(fn):
    """Open file with name `fn`, return open(fn).read()."""
    fh = open(fn, 'r')
    txt = fh.read()
    fh.close()
    return txt

#-----------------------------------------------------------------------------

def pipe_to_file(f, txt):
    """Print string `txt` to file `f`. Close if `f` is a file object."""
    fh = fileo(f, 'w')
    print >>fh, txt 
    fh.close()

#-----------------------------------------------------------------------------

def file_readlines(fn):
    """Open file with name `fn`, return open(fn).readlines()."""
    fh = open(fn, 'r')
    lst = fh.readlines()
    fh.close()
    return lst
    
