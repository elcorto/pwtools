import numpy as np

class Factory(object):
    def __init__(self, array_comp_func=None):
        self.array_comp_func = array_comp_func
    
    def __call__(self, d1, d2):
        arr_t = type(np.array([1.0]))
        for key, val in d1.iteritems():
            assert d2.has_key(key), "dict d2 missing key: %s" %str(key)
            if type(val) == arr_t:
                self.array_comp_func(d1[key], d2[key])
            else:
                assert d1[key] == d2[key]

def assert_attrs_not_none(pp, attr_lst=None, none_attrs=[]):
    lst = pp.attr_lst if attr_lst is None else attr_lst
    for attr_name in lst:
        print "testing attr:", attr_name
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: obj: %s attr: %s is None" \
                %(str(pp), attr_name)


aaae = np.testing.assert_array_almost_equal
aae = np.testing.assert_almost_equal

assert_dict_with_arrays_almost_equal = Factory(aaae)
assert_dict_with_arrays_equal = Factory(lambda x,y: (x==y).all())

adae = assert_dict_with_arrays_almost_equal
ade = assert_dict_with_arrays_equal


