from pwtools.parse import PwSCFOutputFile
from pwtools import common

def test():
    filename = 'files/pw.scf.out'
    common.system('gunzip %s.gz' %filename)
    pp = PwSCFOutputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)
    none_attrs = []
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
    assert pp.scf_converged is True
    common.system('gzip %s' %filename)
