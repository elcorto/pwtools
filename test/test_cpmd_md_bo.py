import os.path
from pwtools.parse import CpmdMDOutputFile
from pwtools import common

def test():
    basedr = 'files/cpmd'
    dr = 'files/cpmd/md_bo'
    common.system('tar -C %s -xzf %s.tgz' %(basedr, dr))
    filename = os.path.join(dr, 'cpmd.bo.out')
    pp = CpmdMDOutputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)
    none_attrs = ['pressure',
                  'stresstensor']
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
    common.system('rm -r %s' %dr)
