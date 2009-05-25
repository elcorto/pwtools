import types

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
    def wrapper(*args):
        largs = list(args)
        if isinstance(largs[0], types.FileType):
            return func(*args)
        elif isinstance(largs[0], types.StringType):
            largs[0] = open(largs[0], 'r')
            ret = func(*tuple(largs))
            largs[0].close()
            return ret         
        else:
            raise ValueError("illegal arg type of '%s', expect file object or "
                             "filename"%repr(largs[0]))
    _doc = func.__doc__            
    _doc = _doc.replace('@args_fh_extra_doc@', "If fh is a file object, it "
        "will not be closed.")
    wrapper.__doc__ = _doc
    wrapper.__name__ = func.__name__            
    return wrapper        
