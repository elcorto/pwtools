import types
import gzip
import textwrap
import os

def open_and_close(func):
    """Decorator for all parsing functions that originally got a file name and
    went thru the whole file b/c the things that they search can occur in
    arbitrary order.
    These functions did
        * open the file
        * go thru
        * close file
    Now, they all expect file objects, as first argument. This decorator
    assures that the first arg to `func` is a file object. 
    Cases:
    1st arg is a fileobject
        do nothig, just call function
    1st arg is a file name
        * open file
        * call func
        * close file
    
    example
    -------
    @open_and_close
    def file_txt_content(fh):
        # fh is a file object
        return fh.read()
    
    >>> fh = open('my_file.txt')
    >>> print file_txt_content(fh)
    >>> fh.close()
    >>>
    >>> print file_txt_content('my_file.txt')
    """
    def wrapper(*args, **kwargs):
        largs = list(args)
        if isinstance(largs[0], types.StringType):
            # Filename case.
            fn = largs[0]
            if fn.endswith('.gz'):
                _open = gzip.open
            else:
                _open = open
            fd = _open(fn, 'r')
            # Files opened with gzip don't have a 'name' attr.
            if not hasattr(fd, 'name'):
                fd.name = os.path.abspath(os.path.expanduser(fn))
            largs[0] = fd
            ret = func(*tuple(largs), **kwargs)
            largs[0].close()
            return ret
        else:
            # File object case. Don't explicitly test for types.FileType b/c
            # that does not if largs[0] is actually a [c]StringIO.StringIO 
            # instances. 
            #
            # Also, the 'name' attribute can be set (largs[0].name = ...) for
            # StringIO.StringIO, but NOT for cStringIO.StringIO. We don't even
            # try fiddling with try-except here. There just won't be any
            # filename.
            return func(*args, **kwargs)
    # TODO: Use functools.wraps() to decorate wrapper() to pass func.__doc__
    # and func.__name__ along.
    _doc = func.__doc__
    if _doc is not None:
        _doc = _doc.replace('@args_fh_extra_doc@', "If fh is a file object, it "
            "will not be closed.")
    wrapper.__doc__ = _doc
    wrapper.__name__ = func.__name__            
    return wrapper        


def add_func_doc(func, doc_func=None):
    """Add the docstring of the function object `doc_func` to func's doc
    string."""
    func.__doc__ += '\n\n' + textwrap.dedent(doc_func.__doc__)
    return func


def crys_add_doc(func):
    """Decorator to add common docstrings to functions with crystal/unit cell
    related functionallity."""
    dct = {}
    dct['cell_doc'] = \
    """cell: array, shape (3,3)
        Matrix with basis vectors as rows."""
    dct['cryst_const_doc'] = \
    """cryst_const: array_like, shape (6,)
        [a, b, c, alpha, beta, gamma]"""
    dct['celldm'] = \
    """celldm: array_like, shape (6,)
        [a, b/a, c/a, cos(alpha), cos(beta), cos(gamma)]
        `a` is supposed to be in Bohr"""
    dct['notes_cell_crys_const'] = \
    """We use PWscf notation.
    CELL_PARAMETERS == (matrix of) primitime basis vectors elsewhere
    crystallographic constants a,b,c,alpha,beta,gamma == cell parameters 
        elsewhere"""
    # Use dictionary string replacement:
    # >>> '%(lala)i %(xxx)s' %{'lala': 3, 'xxx': 'grrr'}
    # '3 grrr'
    func.__doc__ = func.__doc__ % dct 
    return func


