import numpy as np
import spglib

from pwtools import atomic_data
from pwtools.crys import Structure


def is_same_struct(st1, st2):
    """Test if two :class:`~pwtools.crys.Structure` instances are the same.

    Use ``numpy.allclose`` for float-type properties.
    """
    # maybe add early stopping (return if the first test fails), start with
    # cheap tests, finally do spacegroup; only if we call this function very
    # often and speed becomes an issue
    ret = True
    same = ['symbols', 'natoms']
    close = ['volume', 'cryst_const', 'coords_frac']
    for attr in same:
        ret = ret and getattr(st1, attr) == getattr(st2, attr)
    for attr in close:
        ret = ret and np.allclose(getattr(st1, attr),
                                  getattr(st2, attr))
    ret = ret and spglib_get_spacegroup(st1) == spglib_get_spacegroup(st2)
    return ret


def spglib2struct(tup):
    """Transform returned tuple from various spglib functions to
    :class:`~pwtools.crys.Structure`.

    This applies to ``spglib.find_primitive()`` and probably some more. They
    return a tuple `(cell,coords_frac,znucl)`. `znucl` is a list of integers
    with atomic core charge (e.g. 1 for H), see
    :data:`pwtools.atomic_data.numbers`.

    Parameters
    ----------
    tup : tuple (3,)
        Return value from ``spglib.find_primitive()`` and maybe others.

    Returns
    -------
    :class:`~pwtools.crys.Structure`
    """
    assert isinstance(tup, tuple)
    assert len(tup) == 3
    symbols = [atomic_data.symbols[ii] for ii in tup[2]]
    st = Structure(coords_frac=tup[1], cell=tup[0], symbols=symbols)
    return st


def struct2spglib(st):
    """Transform :class:`~pwtools.crys.Structure` to spglib input tuple (cell,
    coords_frac, znucl).
    """
    return st.get_spglib()


def spglib_get_primitive(struct, **kwds):
    """Find primitive structure for given :class:`~pwtools.crys.Structure`.

    If `struct` is irreducible (is already a primitive cell), we return None,
    else a Structure.

    Uses spglib.

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
    spglib used to return (None,None,None) if no primitive cell can be found,
    i.e. the given input Structure cannot be reduced, which can occur if (a) a
    given Structure is already a primitive cell or (b) any other reason like a
    too small value of `symprec`. Now [py]spglib >= 1.8.x seems to always
    return data instead. We use :func:`is_same_struct` to determine if the
    struct is irreducible. In that case we return None in order to keep the API
    unchanged.

    Also note that a primitive cell (e.g. with 2 atoms) can have a number of
    different realizations. Therefore, you may not always get the primitive
    cell which you would expect or get from other tools like Wien2K's sgroup.
    Only things like `natoms` and the spacegroup can be safely compared.
    """
    candidate = spglib2struct(spglib.find_primitive(struct.get_spglib(),
                                                    **kwds))
    if is_same_struct(candidate, struct):
        return None
    else:
        return candidate


def spglib_get_spacegroup(struct, **kwds):
    """Find spacegroup for given Structure.

    Uses spglib.

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
    ret = spglib.get_spacegroup(struct.get_spglib(), **kwds)
    spl = ret.split()
    spg_sym = spl[0]
    spg_num = spl[1]
    spg_num = spg_num.replace('(','').replace(')','')
    spg_num = int(spg_num)
    return spg_num,spg_sym
