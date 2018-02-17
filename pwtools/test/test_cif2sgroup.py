from pwtools import common
from pwtools.test import tools

def test_cif2sgroup():
    tools.skip_if_pkg_missing('CifFile')
    print(common.backtick("../bin/cif2sgroup.py files/cif_struct.cif 2>/dev/null"))
