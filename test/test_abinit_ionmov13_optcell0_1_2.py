from pwtools.parse import AbinitVCMDOutputFile
from pwtools import common

def check(pp, none_attrs=[]):
    for attr_name in pp.attr_lst:
        print("    attr: %s" %attr_name)
##        print  getattr(pp, attr_name)
        if not attr_name in none_attrs:
            assert getattr(pp, attr_name) is not None

def test():
    
    filename = 'files/abi_ionmov13_optcell2.out'
    print("testing: %s" %filename)
    common.system('gunzip %s.gz' %filename)
    pp = AbinitVCMDOutputFile(filename=filename)
    pp.parse()
    check(pp)
    common.system('gzip %s' %filename)
    
    # output file should have exactly the same form of that for optcell 2
    filename = 'files/abi_ionmov13_optcell1.out'
    print("testing: %s" %filename)
    common.system('gunzip %s.gz' %filename)
    pp = AbinitVCMDOutputFile(filename=filename)
    pp.parse()
    check(pp)
    common.system('gzip %s' %filename)

    filename = 'files/abi_ionmov13_optcell0.out'
    print("testing: %s" %filename)
    common.system('gunzip %s.gz' %filename)
    pp = AbinitVCMDOutputFile(filename=filename)
    pp.parse()
    none_attrs = ['acell',
                  'cell',
                  'volume',
                  'lengths',
                  'angles',
                  'rprim',
                  'cryst_const']
    check(pp, none_attrs=none_attrs)
    common.system('gzip %s' %filename)
