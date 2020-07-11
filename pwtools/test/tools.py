"""numpy.testing like functions for usage in tests. We also have tools to
compare nested dictionaries containing numpy arrays etc.

The following functions are defined:

    | :func:`array_equal`
    | :func:`all_types_no_dict_equal`
    | :func:`dict_with_all_types_equal`
    | :func:`all_types_equal`
    |
    | :func:`array_almost_equal`
    | :func:`all_types_no_dict_almost_equal`
    | :func:`dict_with_all_types_almost_equal`
    | :func:`all_types_almost_equal`

For each, we also have a corresponding ``assert_*`` function.

The only high-level functions which you really need are :func:`all_types_equal`
and :func:`all_types_almost_equal`. These also handle nested dicts with numpy
arrays etc. All other ``*equal()`` function are used by those for special
cases, but can also be called directly, of course.

How to change pre-defined comparison functions
----------------------------------------------
::

    >>> import pwtools.test.tools as tt
    >>> # Update comparison functions's comp_map dictionary
    >>> tt.all_types_almost_equal.comp_map[tt.arr_t] =
    ... lambda x,y: tt.true_or_false(np.allclose(x,y,atol=0.1,rtol=0.1))
    >>> tt.assert_all_types_almost_equal.comp_func.comp_map[tt.arr_t] =
    ... lambda x,y: tt.true_or_false(np.allclose(x,y,atol=0.1,rtol=0.1))


Easy, eh? :)
"""

import warnings, copy, tempfile, os, importlib
import numpy as np
from pwtools import num, common
from pwtools.test.testenv import testdir
from nose.plugins.skip import SkipTest
##warnings.simplefilter('always')

#-----------------------------------------------------------------------------
# define types, could probably also use the types module
#-----------------------------------------------------------------------------

arr_t = type(np.array([1.0]))
dict_t = type({'1': 1})
float_t = type(1.0)
np_float_t = type(np.array([1.0])[0])
int_t = type(1)
np_int_t = type(np.array([1])[0])


#-----------------------------------------------------------------------------
# helper functions
#-----------------------------------------------------------------------------

def msg(txt):
    """Uncomment for debugging if tests fail."""
    print(txt)
##    pass


def err(txt):
    print(('error: ' + txt))


def true_or_false(cond):
    """Wrapper for a fucntion which returns bool. Should be used to build all
    comp funcs."""
    if cond:
        print(".. ok")
        return True
    else:
        print(".. uuhhhh, FAIL!")
        return False

#-----------------------------------------------------------------------------
# Factory classes to build more complex comp funcs.
#-----------------------------------------------------------------------------

class AllTypesFactory:
    """Factory for creating functions which compare "any" type.

    We need a dict `comp_map` which maps types (result of ``type(foo)``) to a
    comparison function for that type which returns True or False and is called
    like ``comp_map[<type>](a,b)``. Also a ``comp_map['default']`` entry is
    needed. This is used for all cases where no entry ``comp_map[<type>]`` for
    a given type can be found.
    """
    def __init__(self, comp_map={}):
        """
        Parameters
        ----------
        comp_map : dict
            Dict with types and comparison fucntions.
            Example: ``{type(np.array([1.0])): np.allclose,
                        type(1): lambda x,y: x==y,
                        'default': lambda x,y: x==y}``
        """
        self.comp_map = comp_map

    def __call__(self, d1, d2, strict=False, **kwds):
        """
        Parameters
        ----------
        d1, d2 : any type
            Things to compare.
        strict : bool
            Force equal types. Then 1.0 and 1 are not equal.
        kwds :
            keywords passed directly to comp func, only useful if you know that
            the used comp func(s) will accept these, else you need to re-define
            the comp func
        """
        d1_t = type(d1)
        d2_t = type(d2)
        if strict:
            if d1_t != d2_t:
                err("AllTypesFactory: d1 (%s) and d2 (%s) are not the "
                      "same type" %(d1_t, d2_t))
                return False
        for typ, comp_func in self.comp_map.items():
            if d1_t == typ:
                msg("AllTypesFactory: type=%s, comp_func=%s" \
                      %(str(typ), str(comp_func)))
                return comp_func(d1, d2, **kwds)
        if 'default' in self.comp_map:
            comp_func = self.comp_map['default']
        else:
            raise Exception("no default comparison function defined, "
                "cannot process type=%s" %(d1_t))
        msg("AllTypesFactory: type=default, comp_func=%s" \
              %str(comp_func))
        return comp_func(d1, d2, **kwds)


class DictWithAllTypesFactory:
    """Factory for creating functions which can compare dicts with values of
    "any" type, also numpy arrays. Nested dicts are possible."""
    def __init__(self, comp_func=None):
        self.comp_func = comp_func

    def __call__(self, d1, d2, keys=None, strict=False, attr_lst=None, **kwds):
        """
        Parameters
        ----------
        d1, d2 : dicts
        keys : sequence of dict keys, optional
            Compare only d1[key] and d2[key] for key in keys. Else compare all
            entries.
        strict : bool
            Force equal types in each dict value. Then 1.0 and 1 are not equal.
        kwds :
            keywords passed directly to comp func, only useful if you know that
            the used comp func(s) will accept these, else you need to re-define
            the comp func
        """
        if attr_lst is not None:
            warnings.warn("'attr_lst' keyword deprecated. Use 'keys' instead.",
                          DeprecationWarning)
            keys = attr_lst
        # Test equal keys only if user doesn't provide them.
        if keys is None:
            d1_keys = list(d1.keys())
            d2_keys = list(d2.keys())
            if len(d1_keys) != len(d2_keys):
                err("DictWithAllTypesFactory: key list not equally long")
                return False
            if set(d1_keys) != set(d2_keys):
                err("DictWithAllTypesFactory: keys not equal")
                return False
        _keys = d1_keys if keys is None else keys
        ret = True
        for key in _keys:
            msg("DictWithAllTypesFactory: testing key=%s" %key)
            d1_t = type(d1[key])
            d2_t = type(d2[key])
            if strict:
                if d1_t != d2_t:
                    err("DictWithAllTypesFactory: d1[%s] (%s) and d2[%s] (%s) are not "
                          "the same type" %(key, d1_t, key, d2_t))
                    return False
            if d1_t == dict_t:
                msg("  DictWithAllTypesFactory: case: dict, recursion")
                ret = ret and self(d1[key], d2[key])
            else:
                msg("  DictWithAllTypesFactory: case: something else, "
                      "comp_func=%s" %str(self.comp_func))
                ret = ret and self.comp_func(d1[key], d2[key], **kwds)
        return ret


class AssertFactory:
    """Factory for comparison functions which simply do ``assert
    comp_func(*args, **kwds)``."""
    def __init__(self, comp_func=None):
        self.comp_func = comp_func

    def __call__(self, *args, **kwds):
        assert self.comp_func(*args, **kwds)


#-----------------------------------------------------------------------------
# Basic comparison functions.
#-----------------------------------------------------------------------------

def default_equal(a, b):
    return true_or_false(a == b)


def array_equal(a,b):
    return true_or_false((a==b).all() and a.ndim == b.ndim \
                         and a.dtype==b.dtype)


def array_almost_equal(a, b, **kwds):
    return true_or_false(np.allclose(a, b, **kwds) and a.ndim == b.ndim \
                        and a.dtype==b.dtype)


def float_almost_equal(a, b, **kwds):
    return true_or_false(np.allclose(a, b, **kwds))

#-----------------------------------------------------------------------------
# comp maps for AllTypesFactory, without dicts
#-----------------------------------------------------------------------------

comp_map_no_dict_equal = {\
    arr_t: array_equal,
    'default': default_equal,
    }

comp_map_no_dict_almost_equal = {\
    arr_t: array_almost_equal,
    int_t: float_almost_equal,
    np_int_t: float_almost_equal,
    float_t: float_almost_equal,
    np_float_t: float_almost_equal,
    'default': default_equal,
    }

#-----------------------------------------------------------------------------
# Comparison functions bases on factory classes.
#-----------------------------------------------------------------------------

# compare all types, but no dicts
all_types_no_dict_equal = AllTypesFactory(comp_map=comp_map_no_dict_equal)
all_types_no_dict_almost_equal = AllTypesFactory(comp_map=comp_map_no_dict_almost_equal)


# compare dicts with any type as values, dict values cause recusion until a
# non-dict is found, that is the compared with comp_func
dict_with_all_types_equal = DictWithAllTypesFactory(all_types_no_dict_equal)
dict_with_all_types_almost_equal = DictWithAllTypesFactory(all_types_no_dict_almost_equal)

comp_map_equal = copy.deepcopy(comp_map_no_dict_equal)
comp_map_equal[dict_t] = dict_with_all_types_equal
comp_map_almost_equal = copy.deepcopy(comp_map_no_dict_almost_equal)
comp_map_almost_equal[dict_t] = dict_with_all_types_almost_equal

# compare all types, also dicts
all_types_equal = AllTypesFactory(comp_map_equal)
all_types_almost_equal = AllTypesFactory(comp_map_almost_equal)

# convenience shortcuts and backwd compat: assert_foo(a,b) = assert foo(a,b)
assert_dict_with_all_types_equal = AssertFactory(dict_with_all_types_equal)
assert_dict_with_all_types_almost_equal = AssertFactory(dict_with_all_types_almost_equal)
assert_all_types_equal = AssertFactory(all_types_equal)
assert_all_types_almost_equal = AssertFactory(all_types_almost_equal)
assert_array_equal = AssertFactory(array_equal)
assert_array_almost_equal = AssertFactory(array_almost_equal)

# backwd compat
adae = assert_dict_with_all_types_almost_equal
ade = assert_dict_with_all_types_equal
aaae = assert_array_almost_equal
aae = assert_array_equal


def assert_attrs_not_none(pp, attr_lst=None, none_attrs=[]):
    """Assert that ``pp.<attr>`` is not None for all attribute names (strings)
    in ``attr_lst``.

    Parameters
    ----------
    pp : something to run getattr() on, may have the attribute "attr_lst"
    attr_lst : sequence of strings, optional
        Attribute names to test. If None then we try ``pp.attr_lst`` if it
        exists.
    none_attrs : sequence of strings, optional
        attr names which are allowed to be None
    """
    if attr_lst is None:
        if hasattr(pp, 'attr_lst'):
            attr_lst = pp.attr_lst
        else:
            raise Exception("no attr_lst from input or test object 'pp'")
    for name in attr_lst:
        msg("assert_attrs_not_none: testing: %s" %name)
        attr = getattr(pp, name)
        if name not in none_attrs:
            assert attr is not None, "FAILED: obj: %s attr: %s is None" \
                %(str(pp), name)


def unpack_compressed(src, prefix='tmp', testdir=testdir, ext=None):
    """Convenience function to uncompress files/some_file.out.gz into a random
    location. Return the filename "path/to/random_location/some_file.out"
    without ".gz", which can be used in subsequent commands.

    Supported file types: gz, tgz, tar.gz
        gunzip path/to/random_location/some_file.out.gz
        tar -C path/to/random_location -xzf path/to/random_location/some_file.out.tgz

    Other compress formats may be implemented as needed.

    Can also be used for slightly more complex unpack business, see for
    example test_cpmd_md.py.

    Parameters
    ----------
    src : str
        path to the compressed file, i.e. files/some_file.out.gz
    prefix : str, optional
        prefix for mkdtemp(), usually __file__ of the test script
        to identify which test script created the random dir
    testdir : str, optional
        'path/to' in the example above, usually
        ``pwtools.test.testenv.testdir``
    ext : str, optional
        file extension of compressed file ('gz', 'tgz', 'tar.gz'), if None then
        it will be guessed from `src`
    """
    # 'path/to/random_location'
    workdir = tempfile.mkdtemp(dir=testdir, prefix=prefix)
    # 'gz'
    ext = src.split('.')[-1] if ext is None else ext
    # 'some_file.out'
    base = os.path.basename(src).replace('.'+ext, '')
    # path/to/random_location/some_file.out
    filename = '{workdir}/{base}'.format(workdir=workdir, base=base)
    cmd = "mkdir -p {workdir}; cp {src} {workdir}/; "
    if ext == 'gz':
        cmd += "gunzip {filename}.{ext};"
    elif ext in ['tgz', 'tar.gz']:
        cmd += "tar -C {workdir} -xzf {filename}.{ext};"
    else:
        raise Exception("unsuported file format of file: {}".format(src))
    cmd = cmd.format(workdir=workdir, src=src, filename=filename, ext=ext)
    print(common.backtick(cmd))
    assert os.path.exists(filename), "unpack failed: '%s' not found" %filename
    return filename


def skip(msg):
    """Use inside a test function or class. This raises
    nose.plugins.skip.SkipTest and makes the test be skipped. Doesn't work as a
    decorator. If you need a decorator to temporarily disable a test function,
    then use unittest.skip(). The latter is supposed to work on test functions
    and classes (probably unittest.TestCase and derivatives, untested).

    Examples
    --------

    from pwtools.test.tools import skip
    import unittest

    def test_foo():
        if some_error_condition:
            skip("skipping this test b/c of foo")

    def test_bar():
        skip("we're not at the bar, skip ordering beer")
        normal_test_code_here()

    @unittest.skip("disable test b/c we're all out of zonk!")
    def test_baz():
        assert baz == zonk
    """
    raise SkipTest(msg)


def skip_if_pkg_missing(pkgs):
    """Call skip() if package(s)/module(s) are not importable.

    Parameters
    ----------
    pkgs : str or sequence
        module/package name or sequence of names
    """
    def try_import(pkg):
        try:
            importlib.import_module(pkg)
        except ImportError:
            skip("skip test, package or module not found: {}".format(pkg))
    if isinstance(pkgs, str):
        try_import(pkgs)
    elif common.is_seq(pkgs):
        for pkg in pkgs:
            try_import(pkg)
    else:
        raise Exception("input is no str or sequence: {}".format(pkgs))
