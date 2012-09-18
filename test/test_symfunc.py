import numpy as np
from pwtools import crys
from pwtools import _flib, _fsymfunc, symfunc
rand = np.random.rand

def test_symfunc():
    #g4, g5
    coords_frac = np.random.rand(10,3)
    cell = np.identity(3)*3
    struct = crys.Structure(coords_frac=coords_frac,
                            cell=cell)
    natoms = struct.natoms                            
    sf = symfunc.SymFunc(struct, precond=True)
    params_all = sf.get_default_params_all()
    sf.set_params(params_all[4], 4)
    sf.set_params(params_all[5], 5)
    # distances are calculate here again (already in SymFunc) but who cares,
    # it's fast and only a test
    cos_anglesijk = crys.angles(struct, pbc=True, deg=False)

    for what in [4,5]:
        print "g%i ..." %what
        py_func = getattr(sf, 'g%i_py' %what)
        gpy = py_func()
        assert (gpy > 0.0).any(), "reference is all zero"
        
        # API
        print "g45_f API ..."
        gf = sf.g45_f(what=what)
        assert np.allclose(gpy, gf)
        
        # call extention functions explicitely
        f_func1 = getattr(_fsymfunc, 'symfunc_45')
        f_func2 = getattr(_fsymfunc, 'symfunc_45_fast')
        
        print "symfunc_45 ..."
        ret = np.zeros_like(gpy, order='F') * 0.0
        f_func1(sf.distsq, cos_anglesijk, ret, params_all[what], what)
        ret = sf._precond(ret)
        assert np.allclose(gpy, ret)
        
        print "symfunc_45_fast ..."
        ret = np.zeros_like(gpy, order='F') * 0.0
        f_func2(sf.distvecs, sf.dists, ret, params_all[what], what)
        ret = sf._precond(ret)
        assert np.allclose(gpy, ret)

    print "cutfunc ..."
    dists = rand(10,10)*5
    rcut = 2.5
    ret = np.empty_like(dists, order='F')
    _fsymfunc.cutfunc(dists, rcut, ret)
    cf1 = ret
    cf2 = symfunc.cutfunc(dists,rcut)
    assert (cf1 > 0.0).any()
    assert np.allclose(cf1, cf2)
