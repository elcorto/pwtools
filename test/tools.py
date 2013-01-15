import numpy as np
from pwtools import num

arr_t = type(np.array([1.0]))
dict_t = type({'1': 1})

class DictWithArraysFactory(object):
    """Factory for creating functions with compare dicts with numpy arrays as
    values."""
    def __init__(self, array_comp_func=None):
        self.array_comp_func = array_comp_func
    
    def __call__(self, d1, d2, attr_lst=None):
        attr_lst = d1.keys() if attr_lst is None else attr_lst
        for key, val in d1.iteritems():
            if key in attr_lst:
                print "DictWithArraysFactory: testing: %s" %key
                assert d2.has_key(key), "dict d2 missing key: %s" %str(key)
                if type(val) == arr_t:
                    a = d1[key]
                    b = d2[key]
                    print ">>>>>>>>>>>>>>>>>>>>>>>>"
                    print a
                    print "------------------------"
                    print b
                    print "<<<<<<<<<<<<<<<<<<<<<<<<"
                    print num.rms(a-b)
                    print "<<<<<<<<<<<<<<<<<<<<<<<<"
                    assert self.array_comp_func(d1[key], d2[key])
                else:
                    assert d1[key] == d2[key]

def assert_attrs_not_none(pp, attr_lst=None, none_attrs=[]):
    attr_lst = pp.attr_lst if attr_lst is None else attr_lst
    for name in attr_lst:
        print "assert_attrs_not_none: testing:", name
        attr = getattr(pp, name)
        if name not in none_attrs:
            assert attr is not None, "FAILED: obj: %s attr: %s is None" \
                %(str(pp), name)

      
aaae = np.allclose
aae = np.testing.assert_almost_equal

assert_dict_with_arrays_almost_equal = DictWithArraysFactory(aaae)
assert_dict_with_arrays_equal = DictWithArraysFactory(lambda x,y: (x==y).all())

adae = assert_dict_with_arrays_almost_equal
ade = assert_dict_with_arrays_equal

def assert_all_types_equal(aa,bb):
    if type(aa) == arr_t:
       assert (aa == bb).all()
    elif type(aa) == dict_t:   
        assert_dict_with_arrays_equal(aa,bb)
    else:
        assert aa == bb

