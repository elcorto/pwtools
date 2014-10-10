import os.path
from pwtools.parse import CpmdMDOutputFile
from pwtools import common, verbose
from pwtools.test.tools import assert_attrs_not_none, unpack_compressed
verbose.VERBOSE = True
pj = os.path.join

def run(filename, none_attrs=[]):
    # filename = 'files/cpmd/md_bo/cpmd.bo.out'
    # basename = 'cpmd.bo.out'
    # archive  = 'files/cpmd/md_bo.tgz'
    bar = '='*78
    print bar
    print "@@testing: %s" %filename
    print bar
    basename = os.path.basename(filename)
    archive = os.path.dirname(filename) + '.tgz'
    workdir = unpack_compressed(archive)
    pp = CpmdMDOutputFile(filename=pj(workdir, basename))
    pp.parse()
    assert_attrs_not_none(pp, none_attrs=none_attrs)
    traj = pp.get_traj()
    attrs3d = ['coords', 
               'coords_frac', 
               'forces', 
               'cell',
               'stress',
               ]
    for attr_name in traj.attr_lst:
        attr = getattr(traj, attr_name)
        if attr_name not in none_attrs:
            if hasattr(attr, 'ndim'):
                print "%s: ndim: %s, shape: %s" %(attr_name, attr.ndim, attr.shape)
            assert attr is not None, "FAILED - None: %s" %attr_name
            if attr_name in attrs3d:
                assert attr.ndim == 3, "FAILED - not 3d: %s" %attr_name

def test_cpmd_md():
    # For BO-MD w/ ODIIS optimizer, ekin_elec = [0,0,...,0] but not None.
    run(filename='files/cpmd/md_bo_odiis/cpmd.bo.out',
        none_attrs=['stress',
                    'pressure',
                    'ekin_cell', 
##                    'ekin_elec',
                    'temperature_cell',
                    ])
    run(filename='files/cpmd/md_bo_odiis_npt/cpmd.out',
        none_attrs=['forces',
                    'ekin_cell', 
##                    'ekin_elec',
                    'temperature_cell',
                    ])        
    run(filename='files/cpmd/md_bo_lanczos/cpmd.bo.out',
        none_attrs=['stress',
                    'pressure',
                    'ekin_cell', 
                    'ekin_elec',
                    'temperature_cell',
                    ]) 
    run(filename='files/cpmd/md_cp_mttk/cpmd.out',
        none_attrs=['forces',
                    ])
    run(filename='files/cpmd/md_cp_pr/cpmd.out',
        none_attrs=['forces',
                    ])
    run(filename='files/cpmd/md_cp_nvt_nose/cpmd.out',
        none_attrs=['ekin_cell', 
                    'temperature_cell',
                    ])               
    run(filename='files/cpmd/md_cp_nve/cpmd.out',
        none_attrs=['ekin_cell', 
                    'temperature_cell',
                    ])
