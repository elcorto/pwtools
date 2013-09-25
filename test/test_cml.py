def test_cml():
    from pwtools.parse import CMLFile
    from pwtools import common

    pp = CMLFile('files/cml_struct.cml', units={'length': 1.0})
    pp.parse()

    common.print_dct(pp.__dict__)
