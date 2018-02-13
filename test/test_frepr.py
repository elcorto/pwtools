from pwtools.common import frepr

def test_frepr():
    assert frepr(1) == '1'
    assert frepr(1.0) == '1.0000000000000000e+00'
    assert frepr(None) == 'None'
    assert frepr('abc') == 'abc'
