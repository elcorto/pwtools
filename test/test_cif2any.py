from pwtools import common

def test_cif2any():
    print(common.backtick("../bin/cif2any.py files/cif_struct.cif"))
