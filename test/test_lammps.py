import os
import numpy as np
from pwtools import io, common, crys, parse
from pwtools.test import tools
from pwtools.test.testenv import testdir
rand = np.random.rand

# write test
def test_write_lammps():
    st = crys.Structure(coords_frac=rand(20,3),
                        cell=rand(3,3),
                        symbols=['Al']*10+['N']*10)
    # align cell to lammps standard [[x,0,0],...]
    st.coords = None  
    st.cell = None
    st.set_all()
    st_fn = common.pj(testdir, 'lmp.struct')
    io.write_lammps(st_fn, st)
    symbols = common.file_read(st_fn + '.symbols').split()
    assert st.symbols == symbols
    cmd = r"grep -A22 Atoms %s | grep '^[0-9]'" %st_fn
    arr = parse.arr2d_from_txt(common.backtick(cmd))
    assert arr.shape == (20,5)
    assert np.allclose(st.coords, arr[:,2:])

# parse tests
def set_atol(atol=1e-8):
    tools.all_types_almost_equal.comp_map[tools.arr_t] = \
        lambda x,y: tools.true_or_false(np.allclose(x, y, atol=atol))

def run(tgz, skip=[], atol_map={}):
    tgz_path = os.path.dirname(tgz)
    unpack_path = tgz.replace('.tgz','')
    common.system("tar -C {} -xzf {}".format(tgz_path,tgz))
    tr1 = io.read_lammps_md_txt("{}/log.lammps".format(unpack_path))
    tr2 = io.read_lammps_md_dcd("{}/log.lammps".format(unpack_path))
    for name in tr1.attr_lst:
        if name in skip:
            continue
        elif atol_map.has_key(name):
            set_atol(atol_map[name])
        else:
            set_atol()
        x1 = getattr(tr1, name)
        x2 = getattr(tr2, name)
        print name
        tools.assert_all_types_almost_equal(x1, x2) 

def test_parse_nvt():
    run('files/lammps/md-nvt.tgz', 
        skip=['forces'], 
        atol_map={'velocity': 1e-2, 'coords_frac': 1e-6})

def test_parse_npt():
    run('files/lammps/md-npt.tgz', 
        skip=['forces', 'coords_frac', 'velocity'], 
        atol_map={'volume': 1e-4})

def test_parse_vc_relax():
    run('files/lammps/vc-relax.tgz', 
        skip=['forces', 'coords_frac', 'velocity'], 
        atol_map={'volume': 1e-4})
