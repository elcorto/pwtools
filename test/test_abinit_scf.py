from pwtools.parse import AbinitSCFOutputFile
from pwtools import common

def check(pp, none_attrs=[]):
    for attr_name in pp.attr_lst:
        print("    attr: %s" %attr_name)
        if not attr_name in none_attrs:
            assert getattr(pp, attr_name) is not None

def test():
    
    filename = 'files/abi_scf.out'
    print("testing: %s" %filename)
    common.system('gunzip %s.gz' %filename)
    pp = AbinitSCFOutputFile(filename=filename)
    pp.parse()
    check(pp)
    common.system('gzip %s' %filename)
