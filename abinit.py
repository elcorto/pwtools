# abinit.py
#
# Tools for building Abinit input files.

def get_typat(symbols, order):
    """
    args:
    -----
    symbols : sequence of strings
        Atom symbols
    order : dict
        dict mapping atom symbols to atom number on the order of the
        pseudopotentials in the files file
    
    returns:
    --------
    list of ints

    example:
    --------
    >>> get_typat(['Al']*3 + ['N']*2, {'Al':1, 'N':2})
    [1,1,1,2,2]
    """
    return [order[ss] for ss in symbols]
