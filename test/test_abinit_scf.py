import numpy as np
from pwtools.parse import AbinitSCFOutputFile
from pwtools import common, crys

assrt_aae = lambda x,y: np.testing.assert_array_almost_equal(x,y)

def check(pp, none_attrs=[]):
    for attr_name in pp.attr_lst:
        print("    attr: %s" %attr_name)
        if not attr_name in none_attrs:
            assert getattr(pp, attr_name) is not None
    assert pp.scf_converged is True            

def test():
    filename = 'files/abi_scf.out'
    print("testing: %s" %filename)
    common.system('gunzip %s.gz' %filename)
    pp = AbinitSCFOutputFile(filename=filename)
    pp.parse()
    check(pp)
    # check consistency
    assrt_aae(crys.rms(pp.forces), pp.forces_rms)
    assrt_aae(crys.cell2cc(pp.cell), pp.cryst_const)
    assrt_aae(crys.volume_cc(pp.cryst_const), pp.volume)
    common.system('gzip %s' %filename)
