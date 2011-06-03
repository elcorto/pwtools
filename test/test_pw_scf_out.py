from pwtools.parse import PwSCFOutputFile
from pwtools import common

def test():
    filename = 'files/pw.scf.out'
    common.system('gunzip %s.gz' %filename)
    pp = PwSCFOutputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)
    none_attrs = ['nstep', 
                 'ekin',
                 'temperature',
                 'coords',
                 'cell',
                 'volume',
                ]             
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
    ##    if attr is None:
    ##        print ">>> '%s'," %attr_name
    common.system('gzip %s' %filename)
