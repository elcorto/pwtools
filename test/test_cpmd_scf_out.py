import os.path
from pwtools.parse import CpmdSCFOutputFile
from pwtools import common
pj = os.path.join

def test():
    dr = 'files/cpmd'
    fns = [pj(dr, fn) for fn in 'cpmd.scf.out', 'GEOMETRY.scale']
    filename = pj(dr, 'cpmd.scf.out')
    for fn in fns:
        common.system('gunzip %s.gz' %fn)
    pp = CpmdSCFOutputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)
    none_attrs = []
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
    assert pp.scf_converged is True
    for fn in fns:
        common.system('gzip %s' %fn)
