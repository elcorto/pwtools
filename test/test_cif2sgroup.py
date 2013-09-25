from pwtools import common

def test_cif2sgroup():
    print common.backtick("../bin/cif2sgroup.py files/cif_struct.cif")
