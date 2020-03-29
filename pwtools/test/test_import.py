def test_import_io():
    # pwtools.io
    from pwtools import io
    assert hasattr(io, 'write_axsf')

    # std lib io
    import io
    assert 'pwtools' not in io.__file__
    assert hasattr(io, 'TextIOWrapper')

def test_absolute_signal():
    # std lib signal
    from pwtools.common import signal
    assert not hasattr(signal, 'fftsample')
    from pwtools import common
    print(common.backtick('ls'))
