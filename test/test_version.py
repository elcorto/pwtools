

def test():
    from pwtools import version
    from distutils.version import StrictVersion

    assert version.version_str != '0.0.0'
    assert version.version != StrictVersion('0.0.0')
