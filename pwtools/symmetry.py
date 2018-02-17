from pwtools import atomic_data
from pwtools.crys import Structure

# FIXME: apparently, newer versions of spglib do NOT return None when the input
# structure is irreducible. Instead, they return (cell, coords_frac, znucl).
# The cell may have another orientation. Therefore, if we want to check for
# irreducibility, we need to check for equal natoms, spacegroup, as well as
# cell volume, cryst_const (NOT nell) using np.allclose.
#
# But the quesion is: do we use the "we return None if the stuct is
# irreducible" behavior anywhere? If not, then fine, just fix the check below
# and we are good.
# 
# versions
# --------
# $ pip search spglib                                                                                                                                           [NORMAL]
# pyspglib (1.8.3.1)  - This is the pyspglib module.
#   INSTALLED: 1.8.3.1 (latest)
# spglib (1.9.9.44)   - This is the spglib module.
#
# pyspglib was renamed spglib, etc etc

def spglib2struct(tup):
    raise Exception("function not usable, fixme")
    """Transform returned tuple from various spglib functions to Structure.

    This applies to ``spglib.find_primitive()`` and probably some more. Their
    doc string says it returns an ``ase.Atoms`` object, but what it actually
    returns is a tuple `(cell,coords_frac,znucl)`. `znucl` is a
    list of integers with atomic core charge (e.g. 1 for H), see
    :data:`pwtools.atomic_data.numbers`.

    Parameters
    ----------
    tup : tuple (3,)
        Return value from ``spglib.find_primitive()`` and maybe others.
        If ``(None,)*3`` then we return None.

    Returns
    -------
    Structure or None
    """
    assert type(tup) == type((1,))
    assert len(tup) == 3
    if tup == (None,)*3:
        return None
    else:
        symbols = [atomic_data.symbols[ii] for ii in tup[2]]
        st = Structure(coords_frac=tup[1], cell=tup[0], symbols=symbols)
        return st


def spglib_get_primitive(struct, **kwds):
    """Find primitive structure for given Structure.

    Uses pyspglib.

    Parameters
    ----------
    struct : Structure
    **kwds : keywords
        passed to ``spglib.find_primitive()``, e.g. `symprec` and
        `angle_tolerance` last time I checked

    Returns
    -------
    Structure or None

    Notes
    -----
    spglib returns (None,None,None) if no primitive cell can be found, i.e. the
    given input Structure cannot be reduced, which can occur if (a) a given
    Structure is already a primitive cell or (b) any other reason like a too
    small value of `symprec`. Then,  we return None. See also
    :func:`spglib2struct`.

    Also note that a primitive cell (e.g. with 2 atoms) can have a number of
    different realizations. Therefore, you may not always get the primitive
    cell which you would expect or get from other tools like Wien2K's sgroup.
    Only things like `natoms` and the spacegroup can be safely compared.
    """
    # XXX why imports here??
    from pyspglib import spglib
    return spglib2struct(spglib.find_primitive(struct.get_fake_ase_atoms(),
                                               **kwds))


def spglib_get_spacegroup(struct, **kwds):
    """Find spacegroup for given Structure.

    Uses pyspglib.

    Parameters
    ----------
    struct : Structure
    **kwds : keywords
        passed to ``spglib.get_spacegroup()``, e.g. `symprec` and
        `angle_tolerance` last time I checked

    Returns
    -------
    spg_num, spg_sym
    spg_num : int
        space group number
    spg_sym : str
        space group symbol

    Notes
    -----
    The used function ``spglib.get_spacegroup()`` returns a string, which we
    split into `spg_num` and `spg_sym`.
    """
    # XXX why imports here??
    from pyspglib import spglib
    ret = spglib.get_spacegroup(struct.get_fake_ase_atoms(), **kwds)
    spl = ret.split()
    spg_sym = spl[0]
    spg_num = spl[1]
    spg_num = spg_num.replace('(','').replace(')','')
    spg_num = int(spg_num)
    return spg_num,spg_sym

