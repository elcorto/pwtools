import os.path
from pwtools.parse import CpmdMDOutputFile
from pwtools import common

def run(filename, none_attrs=[]):
    # filename = 'files/cpmd/md_bo/cpmd.bo.out'
    # dr       = 'files/cpmd/md_bo'
    # basedr   = 'files/cpmd'
    # archive  = 'files/cpmd/md_bo.tgz'
    bar = '='*78
    print bar
    print "@@testing: %s" %filename
    print bar
    dr = os.path.dirname(filename)
    basedr = os.path.dirname(dr)
    common.system('tar -C %s -xzf %s.tgz' %(basedr, dr))
    pp = CpmdMDOutputFile(filename=filename)
    pp.parse()
    common.print_dct(pp.__dict__)
    for attr_name in pp.attr_lst:
        attr = getattr(pp, attr_name)
        if attr_name not in none_attrs:
            assert attr is not None, "FAILED: %s" %attr_name
    common.system('rm -r %s' %dr)

def test():
    # For BO-MD w/ ODIIS optimizer, ekin_electrons = [0,0,...,0] but not None.
    run(filename='files/cpmd/md_bo_odiis/cpmd.bo.out',
        none_attrs=['pressure', 
                    'stresstensor', 
                    'ekin_cell', 
##                    'ekin_electrons',
                    'temperature_cell',
                    ])
    run(filename='files/cpmd/md_bo_odiis_npt/cpmd.out',
        none_attrs=['forces',
                    'ekin_cell', 
##                    'ekin_electrons',
                    'temperature_cell',
                    ])        
    run(filename='files/cpmd/md_bo_lanczos/cpmd.bo.out',
        none_attrs=['pressure', 
                    'stresstensor',
                    'ekin_cell', 
                    'ekin_electrons',
                    'temperature_cell',
                    ]) 
    run(filename='files/cpmd/md_cp_mttk/cpmd.out',
        none_attrs=['forces'])
    run(filename='files/cpmd/md_cp_pr/cpmd.out',
        none_attrs=['forces'])        
    run(filename='files/cpmd/md_cp_nvt_nose/cpmd.out',
        none_attrs=['ekin_cell', 
                    'temperature_cell',
                    ])               
    run(filename='files/cpmd/md_cp_nve/cpmd.out',
        none_attrs=['ekin_cell', 
                    'temperature_cell',
                    ])
