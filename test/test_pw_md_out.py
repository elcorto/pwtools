from pwtools.parse import PwMDOutputFile
from pwtools import common

def test():    
    filename = 'files/pw.md.out'
    common.system('gunzip %s.gz' %filename)
    pp = PwMDOutputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)
    none_attrs = ['cell',
                  'volume',
                  'scf_converged'
                 ]             
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
        ##if attr is None:
        ##    print ">>> '%s'," %attr_name
    common.system('gzip %s' %filename)
